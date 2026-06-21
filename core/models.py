"""Skema data seragam untuk semua platform.

Satu baris = satu dokumen (artikel berita / tweet / post IG / dst).
Edge interaksi (mention/retweet/reply) disimpan terpisah untuk SNA (Fase 3).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Document:
    platform: str                      # news | twitter | instagram | facebook
    source: str                        # domain media / akun
    url: str
    content: str
    title: str = ""
    author: str = ""
    lang: str = ""
    published_at: str = ""             # ISO8601 bila ada
    collected_at: str = field(default_factory=_utcnow_iso)
    engagement: dict = field(default_factory=dict)   # likes/shares/retweet/dst
    raw: dict = field(default_factory=dict)          # payload mentah
    keywords_matched: list = field(default_factory=list)

    # Diisi tahap analisis
    sentiment_label: str = ""          # positive | neutral | negative
    sentiment_score: Optional[float] = None

    # ID stabil & unik lintas platform (dedup).
    @property
    def doc_id(self) -> str:
        basis = f"{self.platform}|{self.url or self.content[:200]}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    def to_row(self) -> dict:
        d = asdict(self)
        d["doc_id"] = self.doc_id
        d["engagement"] = json.dumps(self.engagement, ensure_ascii=False)
        d["raw"] = json.dumps(self.raw, ensure_ascii=False)
        d["keywords_matched"] = json.dumps(self.keywords_matched, ensure_ascii=False)
        return d
