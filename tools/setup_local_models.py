"""Unduh IndoBERT SEKALI lalu simpan ke folder lokal proyek.

Setelah ini, program memakai model dari `models/indobert-sentiment/` —
tidak lagi bergantung cache HuggingFace atau koneksi internet, dan folder
itu bisa diganti dengan hasil fine-tuning sendiri.

Jalankan:
    python tools/setup_local_models.py            # unduh & simpan
    python tools/setup_local_models.py --check    # cek status saja
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.sentiment import HUB_MODEL, DEFAULT_MODEL_DIR, _is_local_model  # noqa: E402
from analysis.preprocess import (RESOURCE_DIR, DEFAULT_KATA_DASAR_FILE,      # noqa: E402
                                 DEFAULT_SLANG_FILE, DEFAULT_STOPWORD_FILE,
                                 Preprocessor)


def _ukuran(path: str) -> str:
    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs)
    return f"{total / 1024 / 1024:.0f} MB"


def status():
    print("── Status aset lokal ──")
    ok_model = _is_local_model(DEFAULT_MODEL_DIR)
    print(f"IndoBERT lokal : {'ADA' if ok_model else 'BELUM'}  ({DEFAULT_MODEL_DIR})")
    if ok_model:
        print(f"                 ukuran {_ukuran(DEFAULT_MODEL_DIR)}")

    for label, path in (("kata dasar", DEFAULT_KATA_DASAR_FILE),
                        ("slang→baku", DEFAULT_SLANG_FILE),
                        ("stopword  ", DEFAULT_STOPWORD_FILE)):
        ada = os.path.isfile(path)
        print(f"Kamus {label}: {'ADA' if ada else 'BELUM'}  ({path})")

    p = Preprocessor()
    print(f"Stemmer Sastrawi aktif: {p.stemming_active} "
          f"(+{len(p.kata_dasar_custom)} kata dasar tambahan)")
    print(f"Kamus slang termuat   : {len(p.slang)} entri")
    print(f"Stopword termuat      : {len(p.stopwords)} kata")
    return ok_model


def unduh():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        print("ERROR: transformers belum terpasang. Jalankan:")
        print("  uv pip install transformers torch")
        return False

    if _is_local_model(DEFAULT_MODEL_DIR):
        print(f"Model lokal sudah ada di {DEFAULT_MODEL_DIR} — dilewati.")
        print("Hapus foldernya bila ingin mengunduh ulang.")
        return True

    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    print(f"Mengunduh '{HUB_MODEL}' (~500 MB, sekali saja)...")
    try:
        tok = AutoTokenizer.from_pretrained(HUB_MODEL)
        mdl = AutoModelForSequenceClassification.from_pretrained(HUB_MODEL)
        tok.save_pretrained(DEFAULT_MODEL_DIR)
        mdl.save_pretrained(DEFAULT_MODEL_DIR)
    except Exception as e:
        print(f"GAGAL: {e}")
        shutil.rmtree(DEFAULT_MODEL_DIR, ignore_errors=True)
        return False

    print(f"Selesai. Model tersimpan di {DEFAULT_MODEL_DIR} ({_ukuran(DEFAULT_MODEL_DIR)})")
    return True


def main():
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    if "--check" in sys.argv:
        status()
        return
    unduh()
    print()
    status()


if __name__ == "__main__":
    main()
