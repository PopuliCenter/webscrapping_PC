"""Collector X/Twitter via twikit (jalur GRATIS, berbasis login akun).

Strategi anti rate-limit (pelajaran dari masalah lama):
  - Rotasi banyak akun  (secrets/x_accounts.yaml)
  - Cache sesi cookie per akun  -> tak login ulang tiap kali (login = sinyal bot)
  - Proxy per akun (opsional, residential disarankan)
  - Jeda acak antar request
  - Saat 1 akun kena limit -> akun itu DIISTIRAHATKAN (cooldown_minutes) lalu
    pindah ke akun berikutnya. Catatan istirahat disimpan ke berkas, jadi tetap
    berlaku setelah program dijalankan ulang.
  - Paginasi memakai result.next() — mengulang pencarian dari halaman pertama
    hanya memboroskan permintaan dan memancing limit.

Butuh: pip install twikit ; isi secrets/x_accounts.yaml (lihat .example).
twikit asinkron -> dijalankan via asyncio.run() di x_collect.py.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import List, Tuple

import yaml

from core.models import Document
from .base import build_edges, extract_mentions, clean_text
from .kuota import (akun_tersedia, catat_kuota, istirahatkan, muat_json,
                    sisa_istirahat, sisa_kuota)

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


def _catat(kuota_file: str, batas: int, docs: list) -> None:
    """Catat pemakaian harian. Dipanggil di SETIAP jalan keluar collect()."""
    if batas > 0 and docs:
        pakai = catat_kuota(kuota_file, "x", len(docs))
        print(f"  [x/twikit] pemakaian hari ini: {pakai}/{batas} tweet")


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
    cd_menit = float(tcfg.get("cooldown_minutes", 15))
    cd_file = tcfg.get("cooldown_file", "data/x_cooldown.json")
    batas_harian = int(cfg_x.get("daily_limit", 0) or 0)
    kuota_file = cfg_x.get("quota_file", "data/kuota_harian.json")

    # Batas harian: menarik sedikit tapi rutin lebih aman daripada sekali banyak.
    jatah = sisa_kuota(kuota_file, "x", batas_harian)
    if jatah == 0:
        print(f"  [x/twikit] batas harian {batas_harian} tweet sudah tercapai -> dilewati")
        return [], []

    cooldown = muat_json(cd_file)
    acct_idx = akun_tersedia(accounts, cooldown, time.time())
    if acct_idx < 0:
        print(f"  [x/twikit] semua akun masih istirahat "
              f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> dilewati")
        return [], []
    if jatah > 0:
        print(f"  [x/twikit] sisa jatah hari ini: {jatah} tweet")

    docs: List[Document] = []
    edges: list = []
    client = None

    async def ensure_client():
        nonlocal client, acct_idx
        if client is None:
            acct_idx = akun_tersedia(accounts, cooldown, time.time(), acct_idx)
            if acct_idx < 0:
                return False
            acct = accounts[acct_idx]
            print(f"  [x/twikit] pakai akun #{acct_idx + 1}: {acct.get('username')}")
            client = await _client_for(acct, cookies_dir)
        return True

    for query in queries:
        if jatah == 0:                      # jatah harian habis di tengah jalan
            print("  [x/twikit] batas harian tercapai -> query berikutnya dilewati")
            break
        # jatah -1 = tanpa batas; selain itu kuota query dipotong sisa jatah
        batas_query = per_query if jatah < 0 else min(per_query, jatah)
        collected = 0
        result = None                       # halaman aktif; None = mulai dari awal
        while collected < batas_query:
            if not await ensure_client():
                print(f"  [x/twikit] semua akun istirahat "
                      f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> stop")
                _catat(kuota_file, batas_harian, docs)
                return docs, edges
            try:
                if result is None:
                    result = await client.search_tweet(
                        query, product=product, count=min(20, batas_query - collected))
                else:
                    # PENTING: hasilnya harus dipakai. Bila tidak, pencarian
                    # mengulang halaman pertama terus — boros permintaan dan
                    # justru memancing rate-limit.
                    result = await result.next()
                if not result:
                    break
                for tw in result:
                    doc, e = parse_tweet(tw)
                    docs.append(doc)
                    edges.extend(e)
                    collected += 1
                    if collected >= batas_query:
                        break
                await asyncio.sleep(random.uniform(dmin, dmax))
            except TooManyRequests:
                istirahatkan(cd_file, cooldown, accounts, acct_idx, cd_menit)
                print(f"  [x/twikit] akun #{acct_idx + 1} kena limit "
                      f"-> istirahat {cd_menit:.0f} menit, ganti akun")
                client, result, acct_idx = None, None, acct_idx + 1
                continue
            except Exception as e:
                print(f"  [x/twikit] error query '{query}': {e}")
                break
        if jatah > 0:
            jatah -= collected
        print(f"  [x/twikit] '{query}': {collected} tweet")

    _catat(kuota_file, batas_harian, docs)
    return docs, edges
