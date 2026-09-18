"""Deteksi akun bot / buzzer berbasis heuristik perilaku.

Tanpa machine learning — memakai sinyal perilaku yang sulit dipalsukan dan
mudah dijelaskan dalam laporan riset:

  1. Frekuensi posting     — posting per hari sangat tinggi
  2. Konten duplikat       — teks yang sama diulang-ulang
  3. Aktivitas 24 jam      — manusia tidur, bot tidak
  4. Pola username         — deretan angka acak di belakang nama
  5. Rasio non-orisinal    — hampir semua hanya retweet/mention
  6. Posting serentak      — banyak akun berbeda menyebar teks identik
                             dalam waktu berdekatan (coordinated behavior)

Pakai:
    from analysis.bot_detect import analyze_actors, coordinated_clusters
    hasil = analyze_actors("data/monitoring.db")
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from .preprocess import Preprocessor, content_signature

RE_TRAILING_DIGITS = re.compile(r"\d{4,}$")   # user12345678


def _parse_time(*candidates) -> Optional[datetime]:
    """Parser waktu longgar — format antar platform berbeda-beda."""
    for s in candidates:
        if not s:
            continue
        s = str(s).strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        for fmt in ("%a %b %d %H:%M:%S %z %Y",      # format twikit/twitter
                    "%Y%m%dT%H%M%SZ",                # format GDELT
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def _load_docs(db_path: str, platform: Optional[str]) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = ("SELECT author, source, content, published_at, collected_at, raw "
         "FROM documents WHERE 1=1")
    params = []
    if platform:
        q += " AND platform=?"
        params.append(platform)
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def analyze_actors(db_path: str, platform: str = "twitter",
                   min_posts: int = 3, p: Optional[Preprocessor] = None) -> list:
    """Skor bot per akun. -> [{'actor','n_posts','bot_score','flags',...}, ...]"""
    pp = p or Preprocessor(use_stemmer=False)
    rows = _load_docs(db_path, platform)

    per_actor = defaultdict(list)
    for r in rows:
        actor = (r.get("author") or r.get("source") or "").strip()
        if actor:
            per_actor[actor].append(r)

    hasil = []
    for actor, docs in per_actor.items():
        n = len(docs)
        if n < min_posts:
            continue

        times = [t for t in (_parse_time(d.get("published_at"), d.get("collected_at"))
                             for d in docs) if t]
        sigs = [content_signature(d.get("content") or "", pp) for d in docs]

        # 1) frekuensi posting per hari
        posts_per_day = 0.0
        if len(times) >= 2:
            span = (max(times) - min(times)).total_seconds() / 86400
            posts_per_day = n / span if span > 0.04 else float(n)   # <1jam -> pakai n
        # 2) rasio konten duplikat
        dup_ratio = 1 - (len(set(sigs)) / n)
        # 3) sebaran jam aktif
        hour_cov = len({t.hour for t in times}) / 24 if times else 0.0
        # 4) pola username
        digit_tail = bool(RE_TRAILING_DIGITS.search(actor))
        # 5) rasio non-orisinal (teks diawali 'rt ' atau hanya mention)
        non_orig = sum(
            1 for d in docs
            if re.match(r"^\s*rt\b", (d.get("content") or ""), re.I)
            or len(pp.tokens(d.get("content") or "")) < 3
        ) / n

        # ── skor gabungan (0..1) ────────────────────────────────
        s_freq = min(posts_per_day / 50.0, 1.0)      # >=50 post/hari -> maksimum
        s_dup = dup_ratio
        s_hour = max(0.0, (hour_cov - 0.6) / 0.4)    # aktif >60% jam -> mencurigakan
        s_name = 1.0 if digit_tail else 0.0
        s_orig = non_orig

        bot_score = (0.30 * s_freq + 0.30 * s_dup + 0.15 * s_hour
                     + 0.10 * s_name + 0.15 * s_orig)

        flags = []
        if s_freq > 0.5:  flags.append(f"posting sangat tinggi ({posts_per_day:.0f}/hari)")
        if dup_ratio > 0.5: flags.append(f"konten duplikat {dup_ratio:.0%}")
        if s_hour > 0.5:  flags.append(f"aktif {hour_cov:.0%} jam dalam sehari")
        if digit_tail:    flags.append("username berakhiran angka acak")
        if non_orig > 0.7: flags.append(f"non-orisinal {non_orig:.0%}")

        hasil.append({
            "actor": actor,
            "n_posts": n,
            "posts_per_day": round(posts_per_day, 1),
            "dup_ratio": round(dup_ratio, 3),
            "hour_coverage": round(hour_cov, 3),
            "non_original": round(non_orig, 3),
            "bot_score": round(min(bot_score, 1.0), 3),
            "kategori": ("tinggi" if bot_score >= 0.6 else
                         "sedang" if bot_score >= 0.35 else "rendah"),
            "flags": flags,
        })

    hasil.sort(key=lambda x: x["bot_score"], reverse=True)
    return hasil


def coordinated_clusters(db_path: str, platform: str = "twitter",
                         min_actors: int = 3, window_minutes: int = 60,
                         p: Optional[Preprocessor] = None) -> list:
    """Cari teks identik yang disebar banyak akun dalam waktu berdekatan.

    Ini sinyal terkuat adanya kampanye terkoordinasi (buzzer).
    -> [{'signature','n_actors','actors','contoh_teks','rentang_menit'}, ...]
    """
    pp = p or Preprocessor(use_stemmer=False)
    rows = _load_docs(db_path, platform)

    grup = defaultdict(list)
    for r in rows:
        actor = (r.get("author") or r.get("source") or "").strip()
        teks = r.get("content") or ""
        if not actor or len(pp.tokens(teks)) < 4:     # abaikan teks terlalu pendek
            continue
        sig = content_signature(teks, pp)
        grup[sig].append((actor, _parse_time(r.get("published_at"), r.get("collected_at")), teks))

    hasil = []
    for sig, items in grup.items():
        actors = {a for a, _, _ in items}
        if len(actors) < min_actors:
            continue
        times = [t for _, t, _ in items if t]
        rentang = ((max(times) - min(times)).total_seconds() / 60) if len(times) >= 2 else 0
        if times and rentang > window_minutes:
            continue                                   # tersebar terlalu lama
        hasil.append({
            "signature": sig[:12],
            "n_actors": len(actors),
            "n_posts": len(items),
            "actors": sorted(actors)[:20],
            "contoh_teks": items[0][2][:160],
            "rentang_menit": round(rentang, 1),
        })

    hasil.sort(key=lambda x: x["n_actors"], reverse=True)
    return hasil


def bot_actors(db_path: str, platform: str = "twitter",
               threshold: float = 0.6) -> set:
    """Himpunan akun yang dianggap bot — untuk memfilter SNA/sentimen."""
    return {a["actor"] for a in analyze_actors(db_path, platform)
            if a["bot_score"] >= threshold}
