"""Unduh IndoBERT SEKALI lalu simpan ke folder lokal proyek.

Setelah ini, program memakai model dari `models/indobert-sentiment/` —
tidak lagi bergantung cache HuggingFace atau koneksi internet, dan folder
itu bisa diganti dengan hasil fine-tuning sendiri.

Jalankan:
    python tools/setup_local_models.py            # unduh model sentimen
    python tools/setup_local_models.py --emotion  # + model emosi
    python tools/setup_local_models.py --all      # keduanya
    python tools/setup_local_models.py --check    # cek status saja
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.sentiment import HUB_MODEL, DEFAULT_MODEL_DIR, _is_local_model  # noqa: E402
from analysis.emotion import (HUB_MODEL as EMO_HUB,                          # noqa: E402
                              DEFAULT_MODEL_DIR as EMO_DIR)
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
    print(f"IndoBERT sentimen: {'ADA' if ok_model else 'BELUM'}  ({DEFAULT_MODEL_DIR})")
    if ok_model:
        print(f"                   ukuran {_ukuran(DEFAULT_MODEL_DIR)}")
    ok_emo = _is_local_model(EMO_DIR)
    print(f"Model emosi      : {'ADA' if ok_emo else 'BELUM'}  ({EMO_DIR})")
    if ok_emo:
        print(f"                   ukuran {_ukuran(EMO_DIR)}")

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


def unduh(hub_id: str, target_dir: str, nama: str) -> bool:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        print("ERROR: transformers belum terpasang. Jalankan:")
        print("  uv pip install transformers torch")
        return False

    if _is_local_model(target_dir):
        print(f"[{nama}] sudah ada di {target_dir} — dilewati "
              f"(hapus foldernya bila ingin unduh ulang).")
        return True

    os.makedirs(target_dir, exist_ok=True)
    print(f"[{nama}] mengunduh '{hub_id}' (~500 MB, sekali saja)...")
    try:
        tok = AutoTokenizer.from_pretrained(hub_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(hub_id)
        tok.save_pretrained(target_dir)
        mdl.save_pretrained(target_dir)
    except Exception as e:
        print(f"[{nama}] GAGAL: {e}")
        shutil.rmtree(target_dir, ignore_errors=True)
        return False

    print(f"[{nama}] selesai -> {target_dir} ({_ukuran(target_dir)})")
    return True


def main():
    os.makedirs(RESOURCE_DIR, exist_ok=True)
    if "--check" in sys.argv:
        status()
        return

    semua = "--all" in sys.argv
    hanya_emosi = "--emotion" in sys.argv and not semua

    if not hanya_emosi:
        unduh(HUB_MODEL, DEFAULT_MODEL_DIR, "sentimen")
    if hanya_emosi or semua or "--emotion" in sys.argv:
        unduh(EMO_HUB, EMO_DIR, "emosi")

    print()
    status()


if __name__ == "__main__":
    main()
