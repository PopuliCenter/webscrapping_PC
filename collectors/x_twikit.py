"""Collector X/Twitter via twikit (jalur GRATIS, berbasis login akun).

Strategi anti rate-limit (pelajaran dari masalah lama):
  - Rotasi banyak akun  (secrets/x_accounts.yaml)
  - Cache sesi cookie per akun  -> tak login ulang tiap kali (login = sinyal bot)
  - Proxy per akun (opsional, residential disarankan)
  - Jeda acak antar request + backoff saat kena TooManyRequests
  - Saat 1 akun kena limit -> pindah akun berikutnya

Butuh: pip install twikit ; isi secrets/x_accounts.yaml (lihat .example).
twikit asinkron -> dijalankan via asyncio.run() di x_collect.py.
"""
from __future__ import annotations

import asyncio
import os
import random
from typing import List, Tuple

import yaml

from core.models import Document
from .base import build_edges, extract_mentions, clean_text

try:
    from twikit import Client
    from twikit.errors import TooManyRequests
    _HAS_TWIKIT = True
except Exception:
    _HAS_TWIKIT = False
    class TooManyRequests(Exception):  # placeholder agar modul tetap impor
        pass


# ── Parsing (murni, bisa diuji tanpa jaringan) ──────────────────
def parse_tweet(tw) -> Tuple[Document, list]:
    """Ubah objek tweet twikit -> (Document, edges interaksi). Defensif via getattr."""
    g = lambda o, a, d=None: getattr(o, a, d)
    user = g(tw, "user")
    author = g(user, "screen_name", "") or ""
    text = clean_text(g(tw, "full_text", None) or g(tw, "text", "") or "")
    tid = str(g(tw, "id", "") or "")
    url = f"https://x.com/{author}/status/{tid}" if author and tid else ""

    rt = g(tw, "retweeted_tweet")
    qt = g(tw, "quote")
    rt_author = g(g(rt, "user"), "screen_name", "") if rt else ""
    qt_author = g(g(qt, "user"), "screen_name", "") if qt else ""

    doc = Document(
        platform="twitter",
        source=author,
        author=author,
        url=url,
        title="",
        content=text,
        lang=g(tw, "lang", "") or "",
        published_at=str(g(tw, "created_at", "") or ""),
        engagement={
            "favorite": g(tw, "favorite_count", 0),
            "retweet": g(tw, "retweet_count", 0),
            "reply": g(tw, "reply_count", 0),
            "quote": g(tw, "quote_count", 0),
            "view": g(tw, "view_count", 0),
        },
        raw={"tweet_id": tid, "user_id": str(g(user, "id", "") or "")},
    )
    edges = build_edges(
        "twitter", author, doc.doc_id, doc.published_at,
        mentions=extract_mentions(text),
        retweet_of=rt_author or "",
        quote_of=qt_author or "",
    )
    return doc, edges


# ── Akun & sesi ─────────────────────────────────────────────────
def _load_accounts(path: str) -> list:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("accounts", [])


async def _client_for(acct: dict, cookies_dir: str):
    os.makedirs(cookies_dir, exist_ok=True)
    uname = acct.get("username", "")
    proxy = acct.get("proxy") or None
    client = Client("en-US", proxy=proxy)
    cookie_path = os.path.join(cookies_dir, f"{uname}.json")

    if os.path.isfile(cookie_path):
        client.load_cookies(cookie_path)            # pakai sesi tersimpan
        return client
    await client.login(                              # login pertama kali
        auth_info_1=uname,
        auth_info_2=acct.get("email") or None,
        password=acct.get("password", ""),
    )
    client.save_cookies(cookie_path)
    return client


# ── Pengumpulan ─────────────────────────────────────────────────
async def collect(cfg_x: dict, queries: List[str]) -> Tuple[List[Document], list]:
    if not _HAS_TWIKIT:
        print("  [x/twikit] twikit belum terpasang (pip install twikit) -> dilewati")
        return [], []

    tcfg = cfg_x.get("twikit", {})
    accounts = _load_accounts(tcfg.get("accounts_file", ""))
    if not accounts:
        print("  [x/twikit] belum ada akun di secrets/x_accounts.yaml -> dilewati")
        return [], []

    cookies_dir = tcfg.get("cookies_dir", "data/x_cookies")
    per_query = int(cfg_x.get("tweets_per_query", 50))
    product = cfg_x.get("product", "Latest")
    dmin = float(tcfg.get("min_delay_sec", 5))
    dmax = float(tcfg.get("max_delay_sec", 12))

    docs: List[Document] = []
    edges: list = []
    acct_idx = 0
    client = None

    async def ensure_client():
        nonlocal client, acct_idx
        if client is None:
            if acct_idx >= len(accounts):
                return False
            acct = accounts[acct_idx]
            print(f"  [x/twikit] pakai akun #{acct_idx + 1}: {acct.get('username')}")
            client = await _client_for(acct, cookies_dir)
        return True

    for query in queries:
        collected = 0
        while collected < per_query:
            if not await ensure_client():
                print("  [x/twikit] semua akun habis -> stop")
                return docs, edges
            try:
                result = await client.search_tweet(query, product=product,
                                                   count=min(20, per_query - collected))
                if not result:
                    break
                for tw in result:
                    doc, e = parse_tweet(tw)
                    docs.append(doc)
                    edges.extend(e)
                    collected += 1
                await asyncio.sleep(random.uniform(dmin, dmax))
                # halaman berikutnya bila masih kurang
                if collected < per_query:
                    nxt = await result.next()
                    if not nxt:
                        break
            except TooManyRequests:
                print(f"  [x/twikit] akun #{acct_idx + 1} kena limit -> rotasi akun")
                client, acct_idx = None, acct_idx + 1
                continue
            except Exception as e:
                print(f"  [x/twikit] error query '{query}': {e}")
                break
        print(f"  [x/twikit] '{query}': {collected} tweet")

    return docs, edges
