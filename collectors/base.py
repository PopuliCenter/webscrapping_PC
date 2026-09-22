"""Util bersama untuk semua collector."""
from __future__ import annotations

import re
from typing import Iterable


_kw_cache: dict = {}


def singkatan_kapital(kw: str) -> bool:
    """True untuk singkatan HURUF BESAR pendek: AS, PBB, MK, DPR.

    Kata seperti "AS" kalau dicocokkan tanpa peduli huruf besar-kecil akan ikut
    cocok pada kata Indonesia "as" (poros) dan kata Inggris "as" — berita Cak
    Imin dan FIFA ASEAN Cup pun tersaring masuk. Karena itu singkatan kapital
    dicocokkan PERSIS hurufnya. Kata kunci biasa tetap bebas huruf besar-kecil.
    """
    k = kw.strip()
    return 1 < len(k) <= 5 and k.isupper() and k.isalpha()


def _kw_pattern(kw: str):
    """Pola batas-kata, di-cache. Hindari 'ikn' cocok di dalam 'detikNews'."""
    pat = _kw_cache.get(kw)
    if pat is None:
        k = kw.strip()
        # \b longgar untuk frasa berspasi; ketat untuk token tunggal.
        if singkatan_kapital(k):
            pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)")
        else:
            pat = re.compile(r"(?<!\w)" + re.escape(k.lower()) + r"(?!\w)")
        _kw_cache[kw] = pat
    return pat


def match_keywords(text: str, keywords: Iterable[str]) -> list:
    """Kembalikan daftar keyword yang muncul (batas kata).

    Bebas huruf besar-kecil, KECUALI singkatan kapital (lihat
    `singkatan_kapital`) yang harus sama persis.
    """
    if not text:
        return []
    low = text.lower()
    hits = []
    for kw in keywords:
        k = kw.strip()
        if not k:
            continue
        # singkatan kapital dicari di teks ASLI (huruf besar berarti), sisanya
        # di teks yang sudah dikecilkan
        if _kw_pattern(k).search(text if singkatan_kapital(k) else low):
            hits.append(kw)
    return hits


_WS = re.compile(r"\s+")
_HTML = re.compile(r"<[^>]+>")
_MENTION = re.compile(r"@(\w{1,15})")


def extract_mentions(text: str) -> list:
    """Ambil username yang di-mention dari teks (@user)."""
    if not text:
        return []
    seen, out = set(), []
    for m in _MENTION.findall(text):
        u = m.lower()
        if u not in seen:
            seen.add(u)
            out.append(m)
    return out


def build_edges(platform: str, author: str, doc_id: str, created_at: str = "",
                mentions=None, retweet_of: str = "", reply_to: str = "",
                quote_of: str = "") -> list:
    """Bangun daftar edge interaksi terarah author -> target untuk SNA."""
    if not author:
        return []
    edges = []

    def add(dst, etype):
        if dst and dst.lower() != author.lower():
            edges.append({
                "platform": platform, "src_actor": author, "dst_actor": dst,
                "edge_type": etype, "doc_id": doc_id, "created_at": created_at,
            })

    add(retweet_of, "retweet")
    add(reply_to, "reply")
    add(quote_of, "quote")
    for u in (mentions or []):
        add(u, "mention")
    return edges


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = _HTML.sub(" ", s)
    return _WS.sub(" ", s).strip()
