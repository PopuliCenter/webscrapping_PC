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

try:
    import instaloader
    _HAS_IG = True
except Exception:
    _HAS_IG = False


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


def _login(cfg_ig: dict):
    accounts_file = cfg_ig.get("accounts_file", "")
    if not (accounts_file and os.path.isfile(accounts_file)):
        return None
    with open(accounts_file, encoding="utf-8") as f:
        acct = (yaml.safe_load(f) or {}).get("accounts", [{}])[0]
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

    L = _login(cfg_ig)
    if L is None:
        print("  [ig] tak ada sesi/akun valid -> dilewati")
        return [], []

    per_tag = int(cfg_ig.get("posts_per_tag", 30))
    dmin = float(cfg_ig.get("min_delay_sec", 8))
    dmax = float(cfg_ig.get("max_delay_sec", 20))
    docs: List[Document] = []
    edges: list = []

    for tag in hashtags:
        tag = tag.lstrip("#")
        try:
            ht = instaloader.Hashtag.from_name(L.context, tag)
            count = 0
            for post in ht.get_posts():
                doc, e = parse_post(post)
                docs.append(doc)
                edges.extend(e)
                count += 1
                time.sleep(random.uniform(dmin, dmax))   # jeda besar = aman
                if count >= per_tag:
                    break
            print(f"  [ig] #{tag}: {count} post")
        except Exception as e:
            print(f"  [ig] error #{tag}: {e}")
            continue

    return docs, edges
