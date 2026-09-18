"""Simpan model ke folder lokal proyek — SEKALI, lalu jalan tanpa internet.

Memakai tools/unduh_model.py (satu berkas per waktu + RESUME bila koneksi
putus), karena koneksi ke huggingface.co dari jaringan ini sering diputus dan
pengunduh bawaan kadang menggantung.

Jalankan:
    python tools/setup_local_models.py              # model sentimen (bawaan)
    python tools/setup_local_models.py --all        # semua model
    python tools/setup_local_models.py --emotion --sarkasme   # pilih sebagian
    python tools/setup_local_models.py --check      # cek status saja

Pilihan: --sentimen --w11wo --emotion --sarkasme --zeroshot --indobertweet --all
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.sentiment import HUB_MODEL, DEFAULT_MODEL_DIR, _is_local_model  # noqa: E402
from analysis.emotion import (HUB_MODEL as EMO_HUB,                          # noqa: E402
                              DEFAULT_MODEL_DIR as EMO_DIR)
from analysis.sarcasm import HUB_MODEL as SARKAS_HUB, DIR_W11WO as SARKAS_DIR  # noqa: E402
from analysis.zeroshot import ZS_MODELS, ZS_LOKAL                             # noqa: E402
from analysis.hf_models import MODEL_REGISTRY                                # noqa: E402
from analysis.preprocess import (DEFAULT_KATA_DASAR_FILE, DEFAULT_SLANG_FILE,  # noqa: E402
                                 DEFAULT_STOPWORD_FILE, Preprocessor)
from tools.unduh_model import unduh_model                                    # noqa: E402

# (bendera, nama, repo HF, folder lokal, dipakai oleh)
MODELS = [
    ("--sentimen", "sentimen (mdhugol)", HUB_MODEL, DEFAULT_MODEL_DIR,
     "analisis sentimen utama"),
    ("--w11wo", "sentimen (w11wo)", MODEL_REGISTRY["w11wo"]["id"],
     MODEL_REGISTRY["w11wo"]["lokal"], "intensitas 5 tingkat + pembanding"),
    ("--emotion", "emosi", EMO_HUB, EMO_DIR, "tab Emosi"),
    ("--sarkasme", "sarkasme", SARKAS_HUB, SARKAS_DIR, "tab Intent & Sarkasme"),
    ("--zeroshot", "zero-shot (mDeBERTa)", ZS_MODELS["mdeberta"], ZS_LOKAL["mdeberta"],
     "intent zero-shot"),
    ("--indobertweet", "dasar latih (IndoBERTweet)", "indolem/indobertweet-base-uncased",
     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "models", "indobertweet-base"), "model dasar fine-tuning sarkasme"),
]


def _ukuran(path: str) -> str:
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    return f"{total / 1024 / 1024:.0f} MB"


def status():
    print("── Model lokal ──")
    for bendera, nama, _, folder, guna in MODELS:
        ada = _is_local_model(folder)
        ukuran = f" {_ukuran(folder):>7}" if ada else ""
        print(f"  {'✓' if ada else '·'} {nama:<22}{ukuran}  [{bendera}]  — {guna}")

    print("── Kamus lokal ──")
    for label, path in (("kata dasar", DEFAULT_KATA_DASAR_FILE),
                        ("slang→baku", DEFAULT_SLANG_FILE),
                        ("stopword", DEFAULT_STOPWORD_FILE)):
        print(f"  {'✓' if os.path.isfile(path) else '·'} {label}")
    p = Preprocessor()
    print(f"  Sastrawi aktif: {p.stemming_active} (+{len(p.kata_dasar_custom)} kata dasar), "
          f"slang {len(p.slang)}, stopword {len(p.stopwords)}")


def main():
    if "--check" in sys.argv:
        status()
        return
    dipilih = [m for m in MODELS if "--all" in sys.argv or m[0] in sys.argv]
    if not dipilih:                                    # bawaan: sentimen saja
        dipilih = [MODELS[0]]
    for bendera, nama, repo, folder, _ in dipilih:
        if _is_local_model(folder):
            print(f"[{nama}] sudah ada — dilewati")
            continue
        print(f"[{nama}]")
        try:
            unduh_model(repo, folder)
        except Exception as e:
            print(f"[{nama}] GAGAL: {e} — jalankan ulang, unduhan akan dilanjutkan.")
    print()
    status()


if __name__ == "__main__":
    main()
