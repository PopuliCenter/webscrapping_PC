"""Collector berita via GDELT Doc 2.0 API.

Gratis, tanpa API key. Mencari artikel global per kata kunci.
Dok: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""
from __future__ import annotations

import time
from typing import List

import requests

from core.models import Document
from .base import match_keywords, clean_text

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _build_query(keywords: List[str], lang: str) -> str:
    # GDELT: gabungkan keyword dengan OR; frasa dikutip.
    terms = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        terms.append(f'"{kw}"' if " " in kw else kw)
    q = " OR ".join(terms) if terms else ""
    if len(terms) > 1:
        q = f"({q})"
    if lang:
        # kode bahasa GDELT, mis. indonesian
        lang_map = {"id": "indonesian", "en": "english"}
        q += f" sourcelang:{lang_map.get(lang, lang)}"
    return q.strip()


def collect(keywords: List[str], lang: str = "id", timespan: str = "1d",
            max_records: int = 75) -> List[Document]:
    if not keywords:
        return []
    query = _build_query(keywords, lang)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        "timespan": timespan,
        "sort": "datedesc",
    }
    data = None
    for attempt in range(3):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=30,
                             headers={"User-Agent": "monitoring-research/1.0"})
            if r.status_code == 429:           # GDELT membatasi laju; tunggu lalu coba lagi
                wait = 5 * (attempt + 1)
                print(f"  [gdelt] 429, tunggu {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"  [gdelt] gagal (percobaan {attempt + 1}): {e}")
            time.sleep(2)
    if data is None:
        return []

    docs: List[Document] = []
    for art in data.get("articles", []):
        title = clean_text(art.get("title", ""))
        url = art.get("url", "")
        matched = match_keywords(title, keywords)
        docs.append(Document(
            platform="news",
            source=art.get("domain", ""),
            url=url,
            title=title,
            content=title,  # GDELT ArtList hanya beri judul; isi penuh via news_rss/trafilatura
            lang=lang,
            published_at=art.get("seendate", ""),
            keywords_matched=matched or keywords,
            raw={"gdelt": True, "language": art.get("language", "")},
        ))
    return docs
