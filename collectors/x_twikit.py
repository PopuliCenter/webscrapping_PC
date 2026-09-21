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
import json
import os
import random
import time
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


# ── Masa istirahat akun (cooldown) ──────────────────────────────
# Limit X pulih sendiri setelah beberapa menit. Akun yang kena limit
# diistirahatkan, bukan dibuang — dan catatannya disimpan ke berkas supaya
# tetap berlaku walau program ditutup lalu dijalankan lagi.
def _muat_cooldown(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return {str(k): float(v) for k, v in (json.load(f) or {}).items()}
    except Exception:
        return {}


def _simpan_cooldown(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [x/twikit] gagal menyimpan cooldown: {e}")


def akun_tersedia(accounts: list, cooldown: dict, sekarang: float,
                  mulai: int = 0) -> int:
    """Indeks akun pertama (>= `mulai`) yang tidak sedang istirahat; -1 bila nihil."""
    for i in range(mulai, len(accounts)):
        nama = str(accounts[i].get("username", i))
        if cooldown.get(nama, 0) <= sekarang:
            return i
    return -1


def sisa_istirahat(accounts: list, cooldown: dict, sekarang: float) -> float:
    """Menit sampai akun pertama bebas lagi (0 bila tak ada catatan)."""
    waktu = [cooldown.get(str(a.get("username", i)), 0)
             for i, a in enumerate(accounts)]
    tersisa = [w - sekarang for w in waktu if w > sekarang]
    return round(min(tersisa) / 60, 1) if tersisa else 0.0


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

    cooldown = _muat_cooldown(cd_file)
    acct_idx = akun_tersedia(accounts, cooldown, time.time())
    if acct_idx < 0:
        print(f"  [x/twikit] semua akun masih istirahat "
              f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> dilewati")
        return [], []

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
        collected = 0
        result = None                       # halaman aktif; None = mulai dari awal
        while collected < per_query:
            if not await ensure_client():
                print(f"  [x/twikit] semua akun istirahat "
                      f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> stop")
                return docs, edges
            try:
                if result is None:
                    result = await client.search_tweet(
                        query, product=product, count=min(20, per_query - collected))
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
                    if collected >= per_query:
                        break
                await asyncio.sleep(random.uniform(dmin, dmax))
            except TooManyRequests:
                nama = str(accounts[acct_idx].get("username", acct_idx))
                cooldown[nama] = time.time() + cd_menit * 60
                _simpan_cooldown(cd_file, cooldown)
                print(f"  [x/twikit] akun #{acct_idx + 1} kena limit "
                      f"-> istirahat {cd_menit:.0f} menit, ganti akun")
                client, result, acct_idx = None, None, acct_idx + 1
                continue
            except Exception as e:
                print(f"  [x/twikit] error query '{query}': {e}")
                break
        print(f"  [x/twikit] '{query}': {collected} tweet")

    return docs, edges
