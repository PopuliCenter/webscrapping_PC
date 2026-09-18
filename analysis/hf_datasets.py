"""Ambil dataset sentimen Bahasa Indonesia BERLABEL MANUSIA dari HuggingFace.

Kenapa penting: melatih model memakai label keluaran IndoBERT sendiri bersifat
SIRKULAR — model hanya meniru dirinya. Dataset di sini dilabeli manusia,
sehingga fine-tuning menghasilkan model yang benar-benar belajar.

Semua skema (nama kolom & arti label) sudah DIVERIFIKASI langsung ke HuggingFace,
bukan ditebak.

Pakai:
    from analysis.hf_datasets import load_labeled, list_sumber
    teks, label = load_labeled("carant", limit=5000)
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from itertools import islice
from typing import Optional

# ── Daftar dataset terverifikasi ────────────────────────────────
# text_col   : kolom berisi teks
# label_col  : kolom berisi label
# label_map  : pemetaan nilai label -> positive/neutral/negative
#              (None = label sudah berupa teks yang benar)
DATASET_REGISTRY = {
    "carant": {
        "id": "carant-ai/indonesian_sentiment_dataset",
        "text_col": "text",
        "label_col": "label_text",
        "label_map": None,                       # sudah 'positive'/'neutral'/'negative'
        "kelas": 3,
        "baris": 1_030_393,
        "catatan": "Terbesar & paling rapi (3 kelas). Gabungan beberapa sumber "
                   "termasuk IndoNLU SmSA. Disarankan sebagai default.",
    },
    "sepid": {
        "id": "sepidmnorozy/Indonesian_sentiment",
        "text_col": "text",
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},   # terverifikasi dari contoh
        "kelas": 2,
        "baris": 11_324,
        "catatan": "Biner (tanpa netral). Ulasan/kalimat umum.",
    },
}

# Dataset yang SENGAJA tidak dipakai default karena arti labelnya tidak jelas.
DATASET_AMBIGU = {
    "IndonesiaAI/offline-school-sentiment-on-indonesian-twitter":
        "Berbasis Twitter (menarik), tapi label 0/1/2 tidak terdokumentasi dan "
        "tidak bisa disimpulkan aman dari sampel. Pakai lewat load_custom() "
        "hanya bila kamu sudah memastikan sendiri artinya.",
}


def list_sumber() -> list:
    """Daftar dataset siap pakai."""
    return [{"kunci": k, **{x: v[x] for x in ("id", "kelas", "baris", "catatan")}}
            for k, v in DATASET_REGISTRY.items()]


def _normalisasi_label(nilai, label_map) -> Optional[str]:
    if label_map is not None:
        return label_map.get(nilai)
    s = str(nilai).strip().lower()
    return s if s in ("positive", "neutral", "negative") else None


def load_labeled(kunci: str = "carant", limit: Optional[int] = 5000,
                 split: str = "train", seimbang: bool = False) -> tuple:
    """Unduh dataset berlabel -> (teks, label).

    limit dibatasi secara STREAMING supaya tidak menarik sejuta baris.
    seimbang=True menyamakan jumlah tiap kelas (berguna sebelum melatih).
    """
    if kunci not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{kunci}' tidak dikenal. Pilihan: "
                         f"{list(DATASET_REGISTRY)}")
    cfg = DATASET_REGISTRY[kunci]
    return load_custom(cfg["id"], cfg["text_col"], cfg["label_col"],
                       cfg["label_map"], limit, split, seimbang)


def load_custom(dataset_id: str, text_col: str, label_col: str,
                label_map: Optional[dict] = None, limit: Optional[int] = 5000,
                split: str = "train", seimbang: bool = False) -> tuple:
    """Muat dataset HuggingFace apa pun dengan pemetaan yang kamu tentukan."""
    try:
        from datasets import load_dataset
    except Exception:
        raise RuntimeError("Library 'datasets' belum terpasang: uv pip install datasets")

    print(f"[hf] mengunduh '{dataset_id}' (split={split}, limit={limit})...")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    iterator = islice(ds, limit) if limit else ds

    teks, label = [], []
    for row in iterator:
        t = (row.get(text_col) or "").strip()
        l = _normalisasi_label(row.get(label_col), label_map)
        if t and l:
            teks.append(t)
            label.append(l)

    if seimbang and teks:
        teks, label = _seimbangkan(teks, label)

    print(f"[hf] diperoleh {len(teks)} contoh — distribusi: {dict(Counter(label))}")
    return teks, label


def _seimbangkan(teks: list, label: list) -> tuple:
    """Potong tiap kelas ke jumlah kelas terkecil."""
    per_kelas = {}
    for t, l in zip(teks, label):
        per_kelas.setdefault(l, []).append(t)
    n = min(len(v) for v in per_kelas.values())
    out_t, out_l = [], []
    for l, items in per_kelas.items():
        out_t.extend(items[:n])
        out_l.extend([l] * n)
    return out_t, out_l


def simpan_csv(teks: list, label: list, path: str) -> str:
    """Simpan sebagai CSV text,label — siap dipakai finetune_indobert."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(zip(teks, label))
    print(f"[hf] disimpan ke {path} ({len(teks)} baris)")
    return path


def gabung_dengan_db(teks: list, label: list, db_path: str,
                     platform: Optional[str] = None) -> tuple:
    """Gabungkan dataset HF dengan data berlabel dari database lokal.

    CATATAN: label dari DB berasal dari IndoBERT (bukan manusia). Gabungan ini
    berguna untuk ADAPTASI DOMAIN (mengenalkan kosakata datamu), tapi porsi
    dari DB sebaiknya kecil agar tidak mendominasi.
    """
    from analysis.ml_classify import load_dataset_from_db
    t2, l2 = load_dataset_from_db(db_path, platform=platform)
    print(f"[hf] gabung: {len(teks)} (HF, label manusia) + {len(t2)} (DB, label model)")
    return list(teks) + list(t2), list(label) + list(l2)


if __name__ == "__main__":
    print("Dataset siap pakai:\n")
    for d in list_sumber():
        print(f"  {d['kunci']:<8} {d['id']}")
        print(f"           {d['baris']:,} baris · {d['kelas']} kelas")
        print(f"           {d['catatan']}\n")
    print("Sengaja TIDAK didaftarkan (label ambigu):")
    for k, v in DATASET_AMBIGU.items():
        print(f"  {k}\n    {v}\n")
