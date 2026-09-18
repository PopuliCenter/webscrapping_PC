"""Siapkan CSV data latih berlabel manusia untuk fine-tuning (lokal / Colab).

Menghasilkan:
  data/latih_20k.csv        — sentimen, 21.000 contoh seimbang (text,label)
  data/latih_sarkasme.csv   — sarkasme, split resmi dipertahankan (text,label,split)
  data/latih_sarkasme_<sumber>.csv — sarkasme + data tambahan (Reddit, dll.) di split latih

Unduhan memakai percobaan ulang karena koneksi ke HuggingFace kadang terputus.

Jalankan:
    python tools/siapkan_data_latih.py              # keduanya
    python tools/siapkan_data_latih.py sarkasme     # salah satu
    python tools/siapkan_data_latih.py sarkasme-plus reddit argilla
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


# ── Sarkasme DIPERLUAS: gabung sumber lain ke data LATIH saja ────────────
# Validasi & uji tetap split resmi Twitter (w11wo) → hasil bisa dibandingkan
# langsung dengan model lama. Sumber tambahan (semua dari HuggingFace):
SARKASME_EKSTRA = {
    # 14k komentar r/indonesia, label dari tag "/s" penulisnya sendiri. Apache-2.0.
    "reddit": {"repo": "w11wo/reddit_indonesia_sarcastic", "format": "json",
               "berkas": {"train": "data/train.json", "validation": "data/validation.json",
                          "test": "data/test.json"},
               "kolom": "text"},
    # 9,7k tweet politik 2025 dilabeli manual (Argilla, 2 anotator). Tanpa lisensi
    # tertulis; label terlihat berisik → dipakai hanya bila terbukti membantu.
    "argilla": {"repo": "adealvii/sarcasm-indo-crawl-9k", "format": "parquet",
                "berkas": {"train": "data/train-00000-of-00001.parquet"},
                "kolom": "full_text"},
    # 250 tweet sarkas SINTETIS buatan LLM (tanpa contoh bukan-sarkas).
    "sintetis": {"repo": "enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot",
                 "format": "csv",
                 "berkas": {"train": "Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot.csv"},
                 "kolom": "tweet"},
}


def _kunci(teks: str) -> str:
    """Bentuk ringkas teks untuk mendeteksi duplikat lintas dataset."""
    import re
    t = re.sub(r"<[^>]+>|https?://\S+|@\w+|#", " ", str(teks).lower())
    return re.sub(r"[^a-z0-9]+", "", t)


def _unduh_biner(repo: str, path: str) -> bytes:
    """Unduh ke .cache/datasets (sekali saja; resume bila terputus)."""
    from tools.unduh_model import unduh_url
    tujuan = os.path.join(PROJECT_DIR, ".cache", "datasets", repo.replace("/", "__"),
                          path.replace("/", "__"))
    if not os.path.isfile(tujuan):
        unduh_url(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", tujuan)
    with open(tujuan, "rb") as f:
        return f.read()


def _baca_ekstra(nama: str) -> dict:
    """-> {split_asal: [(teks, label)]} untuk satu sumber tambahan."""
    import pandas as pd
    cfg = SARKASME_EKSTRA[nama]
    hasil = {}
    for split, path in cfg["berkas"].items():
        isi = io.BytesIO(_unduh_biner(cfg["repo"], path))
        df = (pd.read_parquet(isi) if cfg["format"] == "parquet" else
              pd.read_json(isi) if cfg["format"] == "json" else pd.read_csv(isi))
        if nama == "argilla":                    # label = jawaban anotator
            df["label"] = df["label_0.responses"].map(
                lambda x: int(x[0]) if x is not None and len(x) else None)
        baris = []
        for t, l in zip(df[cfg["kolom"]], df["label"]):
            if isinstance(t, str) and t.strip() and l is not None and str(l) in ("0", "1"):
                baris.append((t.strip(), SARKASME_PETA[str(int(l))]))
        hasil[split] = baris
    return hasil


def siapkan_sarkasme_plus(sumber: tuple = ("reddit",)) -> str:
    """Tulis data/latih_sarkasme_<sumber>.csv (mis. latih_sarkasme_reddit.csv).

    - train      : Twitter resmi + sumber tambahan (duplikat dibuang)
    - validation : Twitter resmi (dipakai memilih epoch terbaik)
    - test       : Twitter resmi (sama persis dengan latih_sarkasme.csv)
    - test_reddit: uji Reddit resmi — hanya untuk dinilai, tidak ikut dilatih
    Teks yang muncul di validasi/uji dibuang dari data latih (cegah kebocoran).
    """
    dasar = os.path.join(DATA_DIR, "latih_sarkasme.csv")
    if not os.path.isfile(dasar):
        siapkan_sarkasme()
    with open(dasar, encoding="utf-8", newline="") as f:
        baris = [(r["text"], r["label"], r["split"]) for r in csv.DictReader(f)]

    ekstra = {n: _baca_ekstra(n) for n in sumber}
    if "reddit" in ekstra:                       # uji Reddit disisihkan untuk penilaian
        baris += [(t, l, "test_reddit") for t, l in ekstra["reddit"].pop("test", [])]

    terlarang = {_kunci(t) for t, _, s in baris if s != "train"}
    sudah = {_kunci(t) for t, _, s in baris if s == "train"}
    for n, per_split in ekstra.items():
        masuk = buang = 0
        for t, l in (x for isi in per_split.values() for x in isi):
            k = _kunci(t)
            if not k or k in terlarang or k in sudah:
                buang += 1
                continue
            sudah.add(k)
            baris.append((t, l, "train"))
            masuk += 1
        print(f"  + {n:<9} {masuk:>6} baris latih ({buang} duplikat/bocor dibuang)")

    out = os.path.join(DATA_DIR, f"latih_sarkasme_{'_'.join(sumber)}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows(baris)
    for s in ("train", "validation", "test", "test_reddit"):
        c = Counter(l for _, l, sp in baris if sp == s)
        if c:
            print(f"  {s:<12} {dict(c)}")
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
    if hasattr(sys.stdout, "reconfigure"):         # konsol Windows (cp1252) tak kenal ✓
        sys.stdout.reconfigure(errors="replace")
    pilih = sys.argv[1:] or ["sentimen", "sarkasme"]
    if "sarkasme" in pilih:
        print("== Sarkasme ==")
        siapkan_sarkasme()
    if "sarkasme-plus" in pilih:
        # contoh: sarkasme-plus reddit argilla   (bawaan: reddit saja)
        sumber = tuple(a for a in pilih if a in SARKASME_EKSTRA) or ("reddit",)
        print(f"== Sarkasme diperluas: {', '.join(sumber)} ==")
        siapkan_sarkasme_plus(sumber)
    if "sentimen" in pilih:
        print("== Sentimen ==")
        siapkan_sentimen()
