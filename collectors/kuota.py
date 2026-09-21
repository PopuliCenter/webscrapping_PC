"""Rem pengaman bersama untuk collector medsos: istirahat akun & batas harian.

Dua mekanisme, keduanya dicatat ke berkas JSON supaya tetap berlaku setelah
program dijalankan ulang (scheduler sering restart):

  - COOLDOWN  : akun yang kena limit diistirahatkan beberapa menit, lalu dipakai
                lagi. Limit platform biasanya pulih sendiri.
  - KUOTA     : batas jumlah item per hari per platform. Menarik sedikit tapi
                terus-menerus jauh lebih aman daripada menarik banyak sekaligus.

Fungsi di sini murni (tanggal & waktu bisa disuntik) supaya bisa diuji tanpa
jaringan dan tanpa menunggu ganti hari.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date


# ── Berkas ──────────────────────────────────────────────────────
def muat_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def simpan_json(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [kuota] gagal menyimpan {path}: {e}")


# ── Istirahat akun ──────────────────────────────────────────────
def _nama(accounts: list, i: int) -> str:
    return str(accounts[i].get("username", i))


def akun_tersedia(accounts: list, cooldown: dict, sekarang: float,
                  mulai: int = 0) -> int:
    """Indeks akun pertama (>= `mulai`) yang tidak sedang istirahat; -1 bila nihil."""
    for i in range(mulai, len(accounts)):
        if cooldown.get(_nama(accounts, i), 0) <= sekarang:
            return i
    return -1


def sisa_istirahat(accounts: list, cooldown: dict, sekarang: float) -> float:
    """Menit sampai akun pertama bebas lagi (0 bila tak ada yang istirahat)."""
    tersisa = [cooldown.get(_nama(accounts, i), 0) - sekarang
               for i in range(len(accounts))
               if cooldown.get(_nama(accounts, i), 0) > sekarang]
    return round(min(tersisa) / 60, 1) if tersisa else 0.0


def istirahatkan(path: str, cooldown: dict, accounts: list, idx: int,
                 menit: float, sekarang: float = 0.0) -> dict:
    """Catat satu akun sedang istirahat, lalu simpan ke berkas."""
    cooldown[_nama(accounts, idx)] = (sekarang or time.time()) + menit * 60
    simpan_json(path, cooldown)
    return cooldown


# ── Kuota harian ────────────────────────────────────────────────
def sisa_kuota(path: str, kunci: str, batas: int, hari: str = "") -> int:
    """Sisa jatah hari ini. `batas` <= 0 berarti tanpa batas (-1)."""
    if not batas or batas <= 0:
        return -1
    hari = hari or date.today().isoformat()
    catatan = muat_json(path).get(kunci, {})
    dipakai = int(catatan.get("jumlah", 0)) if catatan.get("tanggal") == hari else 0
    return max(0, int(batas) - dipakai)


def catat_kuota(path: str, kunci: str, jumlah: int, hari: str = "") -> int:
    """Tambah pemakaian hari ini (hitungan mulai dari 0 bila hari berganti)."""
    if jumlah <= 0:
        return 0
    hari = hari or date.today().isoformat()
    data = muat_json(path)
    catatan = data.get(kunci, {})
    dipakai = int(catatan.get("jumlah", 0)) if catatan.get("tanggal") == hari else 0
    data[kunci] = {"tanggal": hari, "jumlah": dipakai + int(jumlah)}
    simpan_json(path, data)
    return data[kunci]["jumlah"]
