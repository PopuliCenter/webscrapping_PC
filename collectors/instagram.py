"""Collector Instagram via instaloader (konten PUBLIK).

Realita: IG sangat agresif membatasi. Wajib login, dan scraping hashtag
rawan dibatasi/diblokir. Mitigasi: sesi login di-cache, jeda besar antar
request, batasi jumlah post. Untuk skala besar -> layanan berbayar (Apify/Bright Data).

Butuh: pip install instaloader ; isi secrets/ig_accounts.yaml (lihat .example).
"""
from __future__ import annotations

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
    import instaloader
    from instaloader.exceptions import (ConnectionException, LoginRequiredException,
                                        QueryReturnedForbiddenException,
                                        TooManyRequestsException)
    # Galat yang berarti "IG sedang membatasi akun ini" -> istirahatkan akunnya.
    _LIMIT_IG = (TooManyRequestsException, ConnectionException,
                 QueryReturnedForbiddenException, LoginRequiredException)
    _HAS_IG = True
except Exception:
    _HAS_IG = False
    _LIMIT_IG = ()


def parse_post(post) -> Tuple[Document, list]:
    """Ubah objek Post instaloader -> (Document, edges). Defensif via getattr."""
    g = lambda a, d=None: getattr(post, a, d)
    owner = g("owner_username", "") or ""
    caption = clean_text(g("caption", "") or "")
    shortcode = g("shortcode", "") or ""
    url = f"https://www.instagram.com/p/{shortcode}/" if shortcode else ""

    # mention dari API bila ada, plus dari teks caption
    api_mentions = list(g("caption_mentions", []) or [])
    mentions = list(dict.fromkeys(api_mentions + extract_mentions(caption)))

    date = g("date_utc", None)
    doc = Document(
        platform="instagram",
        source=owner,
        author=owner,
        url=url,
        content=caption,
        lang="",
        published_at=date.isoformat() if date else "",
        engagement={"like": g("likes", 0) or 0, "comment": g("comments", 0) or 0},
        raw={"shortcode": shortcode, "hashtags": list(g("caption_hashtags", []) or [])},
    )
    edges = build_edges("instagram", owner, doc.doc_id, doc.published_at, mentions=mentions)
    return doc, edges


def _load_accounts(path: str) -> list:
    if not (path and os.path.isfile(path)):
        return []
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("accounts", []) or []


def _login(cfg_ig: dict, acct: dict):
    user, pwd = acct.get("username", ""), acct.get("password", "")
    if not user:
        return None

    L = instaloader.Instaloader(download_pictures=False, download_videos=False,
                                download_comments=False, save_metadata=False,
                                quiet=True)
    sess_dir = cfg_ig.get("cookies_dir", "data/ig_sessions")
    os.makedirs(sess_dir, exist_ok=True)
    sess_path = os.path.join(sess_dir, f"{user}.session")
    try:
        if os.path.isfile(sess_path):
            L.load_session_from_file(user, sess_path)      # pakai sesi tersimpan
        else:
            L.login(user, pwd)
            L.save_session_to_file(sess_path)
        return L
    except Exception as e:
        print(f"  [ig] login gagal: {e}")
        return None


def collect(cfg_ig: dict) -> Tuple[List[Document], list]:
    if not _HAS_IG:
        print("  [ig] instaloader belum terpasang (pip install instaloader) -> dilewati")
        return [], []
    hashtags = cfg_ig.get("hashtags") or []
    if not hashtags:
        print("  [ig] tak ada hashtags di config -> dilewati")
        return [], []

    accounts = _load_accounts(cfg_ig.get("accounts_file", ""))
    if not accounts:
        print("  [ig] belum ada akun di secrets/ig_accounts.yaml -> dilewati")
        return [], []

    per_tag = int(cfg_ig.get("posts_per_tag", 30))
    dmin = float(cfg_ig.get("min_delay_sec", 8))
    dmax = float(cfg_ig.get("max_delay_sec", 20))
    cd_menit = float(cfg_ig.get("cooldown_minutes", 60))
    cd_file = cfg_ig.get("cooldown_file", "data/ig_cooldown.json")
    batas_harian = int(cfg_ig.get("daily_limit", 0) or 0)
    kuota_file = cfg_ig.get("quota_file", "data/kuota_harian.json")

    jatah = sisa_kuota(kuota_file, "instagram", batas_harian)
    if jatah == 0:
        print(f"  [ig] batas harian {batas_harian} post sudah tercapai -> dilewati")
        return [], []

    cooldown = muat_json(cd_file)
    idx = akun_tersedia(accounts, cooldown, time.time())
    if idx < 0:
        print(f"  [ig] semua akun masih istirahat "
              f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> dilewati")
        return [], []
    if jatah > 0:
        print(f"  [ig] sisa jatah hari ini: {jatah} post")

    docs: List[Document] = []
    edges: list = []
    L = None

    def siapkan():
        """Login dengan akun yang tidak sedang istirahat. False bila habis."""
        nonlocal L, idx
        while L is None:
            idx = akun_tersedia(accounts, cooldown, time.time(), idx)
            if idx < 0:
                return False
            print(f"  [ig] pakai akun #{idx + 1}: {accounts[idx].get('username')}")
            L = _login(cfg_ig, accounts[idx])
            if L is None:                  # kredensial/sesi bermasalah -> akun lain
                istirahatkan(cd_file, cooldown, accounts, idx, cd_menit)
                idx += 1
        return True

    for tag in hashtags:
        if jatah == 0:
            print("  [ig] batas harian tercapai -> tagar berikutnya dilewati")
            break
        if not siapkan():
            print(f"  [ig] semua akun istirahat "
                  f"(~{sisa_istirahat(accounts, cooldown, time.time())} menit lagi) -> stop")
            break
        tag = tag.lstrip("#")
        batas_tag = per_tag if jatah < 0 else min(per_tag, jatah)
        count = 0
        try:
            ht = instaloader.Hashtag.from_name(L.context, tag)
            for post in ht.get_posts():
                doc, e = parse_post(post)
                docs.append(doc)
                edges.extend(e)
                count += 1
                if count >= batas_tag:
                    break
                time.sleep(random.uniform(dmin, dmax))   # jeda besar = aman
            print(f"  [ig] #{tag}: {count} post")
        except _LIMIT_IG as e:
            # IG membatasi: akun ini diistirahatkan (lebih lama dari X — IG
            # jauh lebih galak), lalu tagar berikutnya memakai akun lain.
            istirahatkan(cd_file, cooldown, accounts, idx, cd_menit)
            print(f"  [ig] akun #{idx + 1} dibatasi ({str(e)[:60]}) "
                  f"-> istirahat {cd_menit:.0f} menit, ganti akun")
            L, idx = None, idx + 1
        except Exception as e:
            print(f"  [ig] error #{tag}: {e}")
        if jatah > 0:
            jatah -= count

    if batas_harian > 0 and docs:
        pakai = catat_kuota(kuota_file, "instagram", len(docs))
        print(f"  [ig] pemakaian hari ini: {pakai}/{batas_harian} post")
    return docs, edges
