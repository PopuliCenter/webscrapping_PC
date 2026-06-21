"""Collector berita dari RSS feed media.

Gratis & stabil. Ambil ringkasan dari RSS; bila fetch_full_text aktif,
coba ekstrak teks artikel penuh (trafilatura -> fallback ringkasan RSS).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse

import feedparser

from core.models import Document
from .base import match_keywords, clean_text

# Ekstraksi artikel penuh bersifat opsional.
try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _published_iso(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return ""


def _full_text(url: str) -> str:
    if not (_HAS_TRAFILATURA and url):
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            txt = trafilatura.extract(downloaded, include_comments=False)
            return clean_text(txt or "")
    except Exception:
        pass
    return ""


def collect(feeds: List[str], keywords: List[str], lang: str = "id",
            fetch_full_text: bool = False) -> List[Document]:
    docs: List[Document] = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  [rss] gagal {feed_url}: {e}")
            continue

        for entry in parsed.entries:
            url = entry.get("link", "")
            title = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", entry.get("description", "")))

            body = summary
            if fetch_full_text:
                full = _full_text(url)
                if full:
                    body = full

            haystack = f"{title} {body}"
            matched = match_keywords(haystack, keywords) if keywords else []
            # Jika ada keyword dikonfigurasi, hanya simpan yang relevan.
            if keywords and not matched:
                continue

            docs.append(Document(
                platform="news",
                source=_domain(url) or _domain(feed_url),
                url=url,
                title=title,
                content=body,
                author=clean_text(entry.get("author", "")),
                lang=lang,
                published_at=_published_iso(entry),
                keywords_matched=matched,
                raw={"feed": feed_url},
            ))
    return docs
