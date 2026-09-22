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


class GdeltGagal(Exception):
    """Permintaan ke GDELT gagal (biasanya 429 = terlalu sering meminta).

    Dibedakan dari "tidak ada artikel": untuk penarikan arsip, jendela yang
    GAGAL harus diulang, sedangkan jendela yang memang kosong tidak.
    """


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


def _minta(params: dict, percobaan: int = 3, jeda: float = 5.0) -> dict:
    """Satu permintaan ke GDELT dengan percobaan ulang. Gagal -> GdeltGagal.

    GDELT membatasi laju cukup ketat; 429 sering muncul. Jeda dinaikkan tiap
    percobaan (5, 10, 15 detik ... dari `jeda`).
    """
    galat = "tidak diketahui"
    for ke in range(1, percobaan + 1):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=40,
                             headers={"User-Agent": "monitoring-research/1.0"})
            if r.status_code == 429:
                galat = "429 (terlalu sering meminta)"
                if ke < percobaan:
                    tunggu = jeda * ke
                    print(f"  [gdelt] 429, tunggu {tunggu:.0f}s...")
                    time.sleep(tunggu)
                continue
            r.raise_for_status()
            return r.json()
        except GdeltGagal:
            raise
        except Exception as e:
            galat = str(e)[:80]
            print(f"  [gdelt] gagal (percobaan {ke}/{percobaan}): {galat}")
            if ke < percobaan:
                time.sleep(jeda)
    raise GdeltGagal(f"gagal setelah {percobaan} percobaan: {galat}")


def collect(keywords: List[str], lang: str = "id", timespan: str = "1d",
            max_records: int = 75, mulai: str = "", akhir: str = "",
            percobaan: int = 3, jeda: float = 5.0, ketat: bool = False) -> List[Document]:
    """Ambil artikel dari GDELT.

    Tanpa `mulai`/`akhir`: berita terbaru sepanjang `timespan`.
    Dengan `mulai`/`akhir` (format YYYYMMDDHHMMSS): ARSIP pada rentang itu —
    dipakai tools/tarik_arsip.py untuk menarik berita lama per kata kunci.
    """
    if not keywords:
        return []
    query = _build_query(keywords, lang)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        "sort": "datedesc",
    }
    if mulai and akhir:
        params["startdatetime"] = mulai
        params["enddatetime"] = akhir
    else:
        params["timespan"] = timespan
    try:
        data = _minta(params, percobaan, jeda)
    except GdeltGagal as e:
        if ketat:
            raise
        print(f"  [gdelt] {e}")
        return []

    docs: List[Document] = []
    for art in data.get("articles", []):
        title = clean_text(art.get("title", ""))
        url = art.get("url", "")
        # Simpan kata kunci yang BENAR muncul saja. Dulu di sini ada fallback
        # "matched or keywords": artikel yang judulnya tak memuat kata kunci apa
        # pun diberi SEMUA kata kunci, sehingga berita nyasar ikut terhitung
        # sebagai topik yang dipantau. Kosong lebih jujur daripada salah label;
        # tools/tarik_arsip.py melabeli ulang setelah isi artikel terambil.
        matched = match_keywords(title, keywords)
        docs.append(Document(
            platform="news",
            source=art.get("domain", ""),
            url=url,
            title=title,
            content=title,  # GDELT ArtList hanya beri judul; isi penuh via news_rss/trafilatura
            lang=lang,
            published_at=art.get("seendate", ""),
            keywords_matched=matched,
            raw={"gdelt": True, "language": art.get("language", "")},
        ))
    return docs
