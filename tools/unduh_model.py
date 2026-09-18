"""Unduh model HuggingFace ke folder lokal — tahan koneksi putus.

Dibuat karena koneksi ke huggingface.co dari jaringan ini sering diputus
(ConnectionReset 10054), sementara pengunduh bawaan kadang menggantung.

Cara kerja:
  - satu berkas satu waktu (paralel memperparah pemutusan)
  - batas waktu tegas: sambung 10 dtk, jeda baca 60 dtk
  - RESUME: kalau putus di tengah, lanjut dari byte terakhir (HTTP Range),
    bukan mulai dari nol — penting untuk berkas ratusan MB
  - ukuran akhir dicocokkan dengan ukuran di server

Jalankan:
    python tools/unduh_model.py <repo_id> <folder_lokal>
    python tools/unduh_model.py MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 models/zeroshot-mdeberta
"""
from __future__ import annotations

import fnmatch
import os
import sys
import time

import requests

# Cukup berkas yang dibutuhkan transformers (PyTorch). Lewati ONNX, .bin ganda, dll.
POLA_DEFAULT = ["config.json", "*.safetensors", "tokenizer*", "vocab*", "merges.txt",
                "spm.model", "sentencepiece*.model", "special_tokens_map.json",
                "added_tokens.json", "README.md", "*results.json"]
TIMEOUT = (10, 60)


def _coba(fungsi, percobaan=15, label=""):
    for a in range(1, percobaan + 1):
        try:
            return fungsi()
        except Exception as e:
            print(f"  ulang {label} ({a}/{percobaan}): {str(e)[:70]}")
            time.sleep(min(3 * a, 20))
    raise RuntimeError(f"Gagal: {label}")


def daftar_berkas(repo: str, pola=None) -> list:
    """-> [(nama_berkas, ukuran_byte)] yang cocok dengan pola."""
    pola = pola or POLA_DEFAULT

    def ambil():
        r = requests.get(f"https://huggingface.co/api/models/{repo}",
                         params={"blobs": "true"}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()["siblings"]

    hasil = []
    for s in _coba(ambil, label="daftar berkas"):
        nama = s["rfilename"]
        if "/" in nama:                      # lewati subfolder (onnx/, dst.)
            continue
        if any(fnmatch.fnmatch(nama, p) for p in pola):
            hasil.append((nama, s.get("size") or 0))
    # model.safetensors ada -> tak perlu pytorch_model.bin
    return hasil


def unduh_berkas(repo: str, nama: str, ukuran: int, folder: str, percobaan: int = 40):
    tujuan = os.path.join(folder, nama)
    sementara = tujuan + ".part"
    if os.path.isfile(tujuan) and (not ukuran or os.path.getsize(tujuan) == ukuran):
        print(f"  ✓ {nama} (sudah ada)")
        return
    url = f"https://huggingface.co/{repo}/resolve/main/{nama}"
    for a in range(1, percobaan + 1):
        sudah = os.path.getsize(sementara) if os.path.isfile(sementara) else 0
        if ukuran and sudah >= ukuran:
            break
        header = {"Range": f"bytes={sudah}-"} if sudah else {}
        try:
            with requests.get(url, headers=header, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 416:          # sudah lengkap
                    break
                r.raise_for_status()
                mode = "ab" if (sudah and r.status_code == 206) else "wb"
                with open(sementara, mode) as f:
                    for potong in r.iter_content(chunk_size=1 << 20):
                        f.write(potong)
            break
        except Exception as e:
            sekarang = os.path.getsize(sementara) if os.path.isfile(sementara) else 0
            persen = f"{sekarang / ukuran:.0%}" if ukuran else f"{sekarang / 1e6:.0f} MB"
            print(f"  putus di {persen} — lanjut ({a}/{percobaan}): {str(e)[:50]}")
            time.sleep(min(2 * a, 15))
    else:
        raise RuntimeError(f"Gagal mengunduh {nama}")

    akhir = os.path.getsize(sementara)
    if ukuran and akhir != ukuran:
        raise RuntimeError(f"{nama}: ukuran {akhir} ≠ {ukuran} di server")
    os.replace(sementara, tujuan)
    print(f"  ✓ {nama} ({akhir / 1e6:.1f} MB)")


def unduh_model(repo: str, folder: str, pola=None) -> str:
    os.makedirs(folder, exist_ok=True)
    berkas = daftar_berkas(repo, pola)
    if any(n == "model.safetensors" for n, _ in berkas):
        berkas = [(n, u) for n, u in berkas if n != "pytorch_model.bin"]
    total = sum(u for _, u in berkas)
    print(f"{repo} -> {folder}  ({len(berkas)} berkas, {total / 1e6:.0f} MB)")
    for nama, ukuran in sorted(berkas, key=lambda x: x[1]):   # kecil dulu
        unduh_berkas(repo, nama, ukuran, folder)
    return folder


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    unduh_model(sys.argv[1], sys.argv[2])
