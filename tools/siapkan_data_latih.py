"""Siapkan CSV data latih berlabel manusia untuk fine-tuning (lokal / Colab).

Menghasilkan:
  data/latih_20k.csv        — sentimen, 21.000 contoh seimbang (text,label)
  data/latih_sarkasme.csv   — sarkasme, split resmi dipertahankan (text,label,split)

Unduhan memakai percobaan ulang karena koneksi ke HuggingFace kadang terputus.

Jalankan:
    python tools/siapkan_data_latih.py              # keduanya
    python tools/siapkan_data_latih.py sarkasme     # salah satu
"""
from __future__ import annotations

import csv
import io
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

SARKASME_REPO = "w11wo/twitter_indonesia_sarcastic"
SARKASME_PETA = {"0": "bukan_sarkas", "1": "sarkas"}   # kartu dataset: 1 = sarcastic


def _unduh_teks_dataset(repo: str, path: str, percobaan: int = 15) -> str:
    """Unduh berkas teks dari repo dataset HF.

    Memakai `requests` dengan batas waktu tegas (sambung 10 dtk, baca 60 dtk):
    di jaringan yang sering memutus koneksi, cara ini tidak bisa menggantung
    selamanya seperti yang kadang terjadi pada hf_hub_download.
    """
    import requests
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    for a in range(1, percobaan + 1):
        try:
            r = requests.get(url, timeout=(10, 60))
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except Exception as e:
            print(f"  ulang {path} ({a}/{percobaan}): {str(e)[:60]}")
            time.sleep(min(3 * a, 20))
    raise RuntimeError(f"Gagal mengunduh {repo}/{path}")


def siapkan_sarkasme() -> str:
    out = os.path.join(DATA_DIR, "latih_sarkasme.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    baris = []
    for split in ("train", "validation", "test"):
        isi = _unduh_teks_dataset(SARKASME_REPO, f"data/{split}.csv")
        for row in csv.DictReader(io.StringIO(isi)):
            t = (row.get("tweet") or "").strip()
            lab = SARKASME_PETA.get(str(row.get("label", "")).strip())
            if t and lab:
                baris.append((t, lab, split))
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows(baris)
    for s in ("train", "validation", "test"):
        print(f"  {s:<10} {dict(Counter(l for _, l, sp in baris if sp == s))}")
    print(f"Tersimpan: {out} ({len(baris)} baris)")
    return out


def siapkan_sentimen(jumlah: int = 21000) -> str:
    from analysis.hf_datasets import load_labeled, simpan_csv
    out = os.path.join(DATA_DIR, "latih_20k.csv")
    teks, label = load_labeled("carant", limit=jumlah * 6, seimbang=True)
    teks, label = teks[:jumlah], label[:jumlah]
    print(f"  distribusi: {dict(Counter(label))}")
    return simpan_csv(teks, label, out)


def muat_split_csv(path: str, split: str) -> tuple:
    """Baca satu split dari CSV berkolom text,label,split."""
    teks, label = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split:
                teks.append(row["text"])
                label.append(row["label"])
    return teks, label


if __name__ == "__main__":
    pilih = sys.argv[1:] or ["sentimen", "sarkasme"]
    if "sarkasme" in pilih:
        print("== Sarkasme ==")
        siapkan_sarkasme()
    if "sentimen" in pilih:
        print("== Sentimen ==")
        siapkan_sentimen()
