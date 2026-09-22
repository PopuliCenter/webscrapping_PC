"""Bobot media: tidak semua pemberitaan setara.

Satu berita di Detik jelas berbeda dampaknya dengan satu berita di blog daerah,
tetapi hitungan volume biasa memperlakukan keduanya sama. Modul ini memberi
BOBOT per media supaya Share of Voice bisa dibaca dua cara:

  - share mentah   : berapa banyak berita   (siapa paling sering ditulis)
  - share berbobot : berapa besar dampaknya (di media sebesar apa)

Daftar tier ada di `resources/media_tier.csv` dan bisa kamu edit kapan saja —
tidak ada model yang perlu dilatih ulang.

BATASNYA: bobot ini perkiraan kasar dari reputasi & jangkauan umum, BUKAN data
oplah/traffic terverifikasi. Ia berguna untuk membandingkan (Detik > blog),
bukan untuk mengklaim "jangkauan 3 juta pembaca".
"""
from __future__ import annotations

import csv
import os
from functools import lru_cache

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_TIER = os.path.join(PROJECT_DIR, "resources", "media_tier.csv")

BOBOT_TIER = {1: 3.0, 2: 2.0, 3: 1.0}      # tier -> bobot
TIER_BAWAAN = 3                             # domain tak terdaftar


@lru_cache(maxsize=1)
def muat_tier(path: str = "") -> dict:
    """-> {domain: tier}. Baris berawalan '#' diabaikan."""
    path = path or FILE_TIER
    peta = {}
    if not os.path.isfile(path):
        return peta
    with open(path, encoding="utf-8", newline="") as f:
        for baris in csv.reader(f):
            if not baris or baris[0].lstrip().startswith("#") or baris[0] == "domain":
                continue
            try:
                peta[baris[0].strip().lower()] = int(baris[1])
            except (IndexError, ValueError):
                continue
    return peta


def domain_induk(source: str) -> str:
    """news.detik.com -> detik.com ; ekonomi.republika.co.id -> republika.co.id.

    Menangani akhiran dua tingkat khas Indonesia (.co.id, .or.id, .web.id).
    """
    s = (source or "").strip().lower().replace("www.", "")
    bagian = s.split(".")
    if len(bagian) <= 2:
        return s
    dua_tingkat = {"co.id", "or.id", "web.id", "go.id", "ac.id", "my.id", "com.au"}
    if ".".join(bagian[-2:]) in dua_tingkat:
        return ".".join(bagian[-3:])
    return ".".join(bagian[-2:])


def tier_media(source: str) -> int:
    peta = muat_tier()
    s = (source or "").strip().lower().replace("www.", "")
    return peta.get(s) or peta.get(domain_induk(s), TIER_BAWAAN)


def bobot_media(source: str) -> float:
    return BOBOT_TIER.get(tier_media(source), 1.0)


def tambah_bobot(df):
    """Tambahkan kolom `tier` & `bobot` ke DataFrame dokumen."""
    if df.empty:
        return df
    d = df.copy()
    d["tier"] = d["source"].map(tier_media)
    d["bobot"] = d["tier"].map(lambda t: BOBOT_TIER.get(t, 1.0))
    return d


def ringkas_tier(df) -> "list[dict]":
    """Sebaran dokumen per tier — untuk melihat mutu sumber pemberitaan."""
    if df.empty:
        return []
    d = tambah_bobot(df)
    keluar = []
    for tier in (1, 2, 3):
        bagian = d[d["tier"] == tier]
        if bagian.empty:
            continue
        keluar.append({
            "tier": tier,
            "dokumen": len(bagian),
            "share_%": round(100 * len(bagian) / len(d), 1),
            "media": ", ".join(sorted(bagian["source"].unique())[:4]),
        })
    return keluar
