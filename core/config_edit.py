"""Ubah daftar di config.yaml TANPA menghapus komentarnya.

`yaml.safe_dump` akan menulis ulang seluruh berkas dan membuang semua komentar
penjelas di config.yaml. Karena itu di sini hanya baris daftar (`- item`) yang
diganti; baris kunci, komentar, dan bagian lain dibiarkan apa adanya.

Setiap penulisan diverifikasi: berkas dibaca ulang dan dibandingkan dengan nilai
yang diminta. Bila tidak cocok, isi lama dikembalikan dan error dilempar —
lebih baik gagal terang-terangan daripada merusak config pengguna.

    set_list("config.yaml", ["x", "search_queries"], ["ikn", "pemilu"])
"""
from __future__ import annotations

import os
import re
from typing import List

import yaml

_ITEM = re.compile(r"^(\s*)-\s")


def _indent(baris: str) -> int:
    return len(baris) - len(baris.lstrip())


def _kosong(baris: str) -> bool:
    s = baris.strip()
    return not s or s.startswith("#")


def _cari_kunci(baris: List[str], path: List[str]) -> int:
    """Nomor baris kunci terakhir pada `path` (mis. ['x','search_queries'])."""
    mulai, batas, induk = 0, len(baris), -1
    for ke, kunci in enumerate(path):
        pola = re.compile(rf"^(\s*){re.escape(kunci)}\s*:")
        for i in range(mulai, batas):
            if _kosong(baris[i]):
                continue
            ind = _indent(baris[i])
            if ind <= induk:                   # keluar dari blok induk
                break
            m = pola.match(baris[i])
            if m and ind == (induk + 2 if induk >= 0 else 0):
                if ke == len(path) - 1:
                    return i
                induk, mulai = ind, i + 1
                # batas blok anak: baris berikutnya dgn indentasi <= induk
                batas = len(baris)
                for j in range(i + 1, len(baris)):
                    if not _kosong(baris[j]) and _indent(baris[j]) <= ind:
                        batas = j
                        break
                break
        else:
            break
    raise KeyError(f"kunci {'.'.join(path)} tidak ditemukan di config")


def set_list(path_yaml: str, path: List[str], nilai: List[str]) -> None:
    """Ganti isi daftar pada `path` dengan `nilai` (string, dikutip ganda)."""
    # newline="" → akhiran baris asli (CRLF di Windows) ikut terbaca dan tetap
    # dipertahankan; tanpa ini seluruh berkas ikut berubah walau isinya sama.
    with open(path_yaml, encoding="utf-8", newline="") as f:
        asli = f.read()
    baris = asli.splitlines(keepends=True)
    eol = "\r\n" if asli.count("\r\n") * 2 > asli.count("\n") else "\n"

    i = _cari_kunci(baris, path)
    ind_kunci = _indent(baris[i])
    akhir = i + 1
    while akhir < len(baris) and _ITEM.match(baris[akhir]) \
            and _indent(baris[akhir]) > ind_kunci:
        akhir += 1
    ind_item = _indent(baris[i + 1]) if akhir > i + 1 else ind_kunci + 2

    bersih = [str(v).strip() for v in nilai if str(v).strip()]
    baru = [f'{" " * ind_item}- "{v}"{eol}' for v in bersih]
    # Baris kunci: daftar kosong ditulis `[]` agar YAML tetap sah; saat diisi
    # lagi, `[]` itu harus dibuang — kalau tidak, item baru diabaikan.
    kunci_teks = baris[i].rstrip("\r\n")
    depan, pisah, komentar = kunci_teks.partition("#")
    kosong_sekarang = depan.rstrip().endswith("[]")
    if kosong_sekarang or not baru:            # selain itu baris kunci tak disentuh,
        depan = depan.rstrip()                 # supaya perataan komentar tetap rapi
        if kosong_sekarang:
            depan = depan[:-2].rstrip()
        if not baru:
            depan += " []"
        baris[i] = depan + (f"   {pisah}{komentar}" if pisah else "") + eol

    isi_baru = "".join(baris[:i + 1] + baru + baris[akhir:])
    with open(path_yaml, "w", encoding="utf-8", newline="") as f:
        f.write(isi_baru)

    try:                                        # verifikasi: hasil harus sama
        cek = yaml.safe_load(open(path_yaml, encoding="utf-8"))
        for k in path:
            cek = cek[k]
        cocok = [str(x) for x in (cek or [])] == bersih
    except Exception:
        cocok = False
    if not cocok:
        with open(path_yaml, "w", encoding="utf-8", newline="") as f:
            f.write(asli)                       # kembalikan isi lama
        raise ValueError(f"gagal menulis {'.'.join(path)} — config dikembalikan")


def baca_list(path_yaml: str, path: List[str]) -> List[str]:
    """Baca daftar pada `path`; kembalikan [] bila tidak ada."""
    if not os.path.isfile(path_yaml):
        return []
    data = yaml.safe_load(open(path_yaml, encoding="utf-8")) or {}
    for k in path:
        if not isinstance(data, dict) or k not in data:
            return []
        data = data[k]
    return [str(x) for x in (data or [])]
