"""Penyimpanan SQLite (stdlib, tanpa dependensi).

Skema seragam -> mudah pindah ke PostgreSQL nanti (tinggal ganti koneksi).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable

from .models import Document

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    platform         TEXT NOT NULL,
    source           TEXT,
    author           TEXT,
    url              TEXT,
    title            TEXT,
    content          TEXT,
    lang             TEXT,
    published_at     TEXT,
    collected_at     TEXT,
    engagement       TEXT,
    raw              TEXT,
    keywords_matched TEXT,
    sentiment_label  TEXT,
    sentiment_score  REAL,
    emotion_label    TEXT,
    emotion_score    REAL,
    intent_label     TEXT,
    intent_score     REAL,
    intensitas_label TEXT,
    intensitas_score REAL,
    sarkasme_label   TEXT,
    sarkasme_score   REAL
);
CREATE INDEX IF NOT EXISTS idx_docs_platform   ON documents(platform);
CREATE INDEX IF NOT EXISTS idx_docs_published  ON documents(published_at);
CREATE INDEX IF NOT EXISTS idx_docs_sentiment  ON documents(sentiment_label);

-- Edge interaksi untuk SNA (Fase 3): siapa -> siapa, lewat dokumen apa.
CREATE TABLE IF NOT EXISTS interactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    platform    TEXT NOT NULL,
    src_actor   TEXT NOT NULL,
    dst_actor   TEXT NOT NULL,
    edge_type   TEXT,                 -- mention | retweet | reply | quote | share
    doc_id      TEXT,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_platform ON interactions(platform);
"""


# Kolom hasil analisis tambahan. Database lama otomatis ditambah kolom ini.
KOLOM_ANALISIS = {
    "emotion_label": "TEXT", "emotion_score": "REAL",
    "intent_label": "TEXT", "intent_score": "REAL",
    "intensitas_label": "TEXT", "intensitas_score": "REAL",
    "sarkasme_label": "TEXT", "sarkasme_score": "REAL",
    # Rincian sentimen per paragraf (berita panjang) — lihat analysis/sentiment.py
    "bagian_total": "INTEGER", "bagian_negatif": "INTEGER",
    "bagian_positif": "INTEGER", "kutipan_negatif": "TEXT",
    # Berita sindikasi: berisi doc_id berita UTAMA bila dokumen ini hanya
    # salinan. Kosong = berita unik. Diisi tools/dedup_sindikasi.py.
    "duplikat_dari": "TEXT",
}


class Storage:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    @staticmethod
    def _migrate(c):
        """Tambah kolom baru pada database lama tanpa kehilangan data."""
        ada = {r[1] for r in c.execute("PRAGMA table_info(documents)").fetchall()}
        for kolom, tipe in KOLOM_ANALISIS.items():
            if kolom not in ada:
                c.execute(f"ALTER TABLE documents ADD COLUMN {kolom} {tipe}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save_documents(self, docs: Iterable[Document]) -> int:
        """Insert idempoten (INSERT OR IGNORE by doc_id). Return jumlah baru."""
        rows = [d.to_row() for d in docs]
        if not rows:
            return 0
        cols = [
            "doc_id", "platform", "source", "author", "url", "title", "content",
            "lang", "published_at", "collected_at", "engagement", "raw",
            "keywords_matched", "sentiment_label", "sentiment_score",
        ]
        placeholders = ",".join("?" for _ in cols)
        sql = f"INSERT OR IGNORE INTO documents ({','.join(cols)}) VALUES ({placeholders})"
        with self._conn() as c:
            before = c.total_changes
            c.executemany(sql, [[r.get(k) for k in cols] for r in rows])
            return c.total_changes - before

    def docs_without_sentiment(self, limit: int = 500):
        with self._conn() as c:
            cur = c.execute(
                "SELECT doc_id, title, content FROM documents "
                "WHERE sentiment_label IS NULL OR sentiment_label = '' LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_sentiment(self, doc_id: str, label: str, score: float):
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET sentiment_label=?, sentiment_score=? WHERE doc_id=?",
                (label, score, doc_id),
            )

    def docs_without_emotion(self, limit: int = 500):
        with self._conn() as c:
            cur = c.execute(
                "SELECT doc_id, title, content FROM documents "
                "WHERE emotion_label IS NULL OR emotion_label = '' LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]

    def update_emotion(self, doc_id: str, label: str, score):
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET emotion_label=?, emotion_score=? WHERE doc_id=?",
                (label, score, doc_id),
            )

    def docs_without_field(self, field: str, limit: int = 500):
        """Dokumen yang kolom analisis `field` masih kosong."""
        if field not in KOLOM_ANALISIS:
            raise ValueError(f"Kolom tidak dikenal: {field}")
        with self._conn() as c:
            cur = c.execute(
                f"SELECT doc_id, title, content FROM documents "
                f"WHERE {field} IS NULL OR {field} = '' LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def update_fields(self, doc_id: str, **fields):
        """Isi beberapa kolom analisis sekaligus (nama kolom divalidasi)."""
        fields = {k: v for k, v in fields.items() if k in KOLOM_ANALISIS}
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE documents SET {sets} WHERE doc_id=?",
                      (*fields.values(), doc_id))

    def save_interactions(self, edges: list) -> int:
        """Simpan edge interaksi (mention/retweet/reply/quote) untuk SNA."""
        if not edges:
            return 0
        cols = ["platform", "src_actor", "dst_actor", "edge_type", "doc_id", "created_at"]
        sql = (f"INSERT INTO interactions ({','.join(cols)}) "
               f"VALUES ({','.join('?' for _ in cols)})")
        with self._conn() as c:
            before = c.total_changes
            c.executemany(sql, [[e.get(k) for k in cols] for e in edges])
            return c.total_changes - before

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            by_plat = {
                r[0]: r[1]
                for r in c.execute(
                    "SELECT platform, COUNT(*) FROM documents GROUP BY platform"
                ).fetchall()
            }
            by_sent = {
                (r[0] or "unlabeled"): r[1]
                for r in c.execute(
                    "SELECT sentiment_label, COUNT(*) FROM documents GROUP BY sentiment_label"
                ).fetchall()
            }
        return {"total": total, "by_platform": by_plat, "by_sentiment": by_sent}
