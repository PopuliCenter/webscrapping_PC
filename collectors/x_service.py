"""Collector X/Twitter via layanan berbayar (fallback STABIL, anti-bot ditangani vendor).

Default: Apify actor (mis. apidojo/tweet-scraper). Bisa diganti vendor lain.
Tidak ada login akun -> tak ada risiko suspend; bayar per data.

Butuh: token API di env (default APIFY_TOKEN). Set di config.yaml -> x.service.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import requests

from core.models import Document
from .base import build_edges, extract_mentions, clean_text


def _g(d: dict, *keys, default=""):
    """Ambil nilai pertama yang ada dari beberapa kemungkinan nama field."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def parse_item(it: dict) -> Tuple[Document, list]:
    """Map satu item dataset (skema vendor bervariasi) -> (Document, edges)."""
    author = str(_g(it, "username", "userName", "author", "screen_name") or "")
    if isinstance(author, dict):
        author = str(author.get("userName", author.get("username", "")))
    text = clean_text(str(_g(it, "text", "full_text", "content")))
    url = str(_g(it, "url", "tweetUrl", "twitterUrl"))
    tid = str(_g(it, "id", "id_str", "tweetId"))

    doc = Document(
        platform="twitter",
        source=author,
        author=author,
        url=url or (f"https://x.com/{author}/status/{tid}" if author and tid else ""),
        content=text,
        lang=str(_g(it, "lang", "language")),
        published_at=str(_g(it, "createdAt", "created_at", "date")),
        engagement={
            "favorite": _g(it, "likeCount", "favorite_count", "likes", default=0),
            "retweet": _g(it, "retweetCount", "retweet_count", "retweets", default=0),
            "reply": _g(it, "replyCount", "reply_count", "replies", default=0),
            "quote": _g(it, "quoteCount", "quote_count", default=0),
            "view": _g(it, "viewCount", "views", default=0),
        },
        raw={"vendor": "service", "tweet_id": tid},
    )
    rt_author = str(_g(it, "retweetedAuthor", "retweeted_screen_name"))
    qt_author = str(_g(it, "quotedAuthor", "quoted_screen_name"))
    edges = build_edges(
        "twitter", author, doc.doc_id, doc.published_at,
        mentions=extract_mentions(text),
        retweet_of=rt_author, quote_of=qt_author,
    )
    return doc, edges


def _run_apify(actor: str, token: str, run_input: dict, timeout: int = 300) -> list:
    """Jalankan actor Apify sinkron & ambil item dataset langsung."""
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
    try:
        r = requests.post(url, params={"token": token}, json=run_input, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [x/service] Apify gagal: {e}")
        return []


def collect(cfg_x: dict, queries: List[str]) -> Tuple[List[Document], list]:
    svc = cfg_x.get("service", {})
    provider = svc.get("provider", "apify").lower()
    token = os.environ.get(svc.get("api_token_env", "APIFY_TOKEN"), "")
    if not token:
        print(f"  [x/service] token env '{svc.get('api_token_env', 'APIFY_TOKEN')}' kosong -> dilewati")
        return [], []

    per_query = int(cfg_x.get("tweets_per_query", 50))
    docs: List[Document] = []
    edges: list = []

    if provider == "apify":
        actor = svc.get("actor", "apidojo~tweet-scraper")
        for query in queries:
            run_input = dict(svc.get("input", {}))     # template input dari config
            run_input.setdefault("searchTerms", [query])
            run_input.setdefault("maxItems", per_query)
            items = _run_apify(actor, token, run_input)
            for it in items:
                if not isinstance(it, dict):
                    continue
                doc, e = parse_item(it)
                docs.append(doc)
                edges.extend(e)
            print(f"  [x/service] '{query}': {len(items)} item")
    else:
        print(f"  [x/service] provider '{provider}' belum didukung")

    return docs, edges
