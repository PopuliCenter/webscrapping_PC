"""Unggah model hasil fine-tuning ke HuggingFace Hub.

⚠️  MENGUNGGAH = MENGIRIM KE LAYANAN EKSTERNAL.
    - Default repo PRIVAT. Publik hanya bila kamu menulis --public secara sadar.
    - Wajib konfirmasi dengan --yes; tanpa itu skrip hanya menampilkan rencana.
    - Pastikan data latihmu tidak memuat informasi pribadi sebelum diunggah;
      bobot model bisa menghafal potongan data latih.

Token: buat di https://huggingface.co/settings/tokens (akses: Write), lalu
    setx HF_TOKEN "hf_xxx"        # Windows (buka terminal baru setelahnya)
    export HF_TOKEN="hf_xxx"      # Linux/Mac

Jalankan:
    python tools/push_to_hub.py --repo namamu/indobert-sentimen-ikn            # pratinjau
    python tools/push_to_hub.py --repo namamu/indobert-sentimen-ikn --yes      # unggah (privat)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOCAL = os.path.join(PROJECT_DIR, "models", "indobert-sentiment-finetuned")

KARTU_MODEL = """---
language: id
license: apache-2.0
tags:
  - sentiment-analysis
  - indonesian
  - indobert
---

# {repo}

Model klasifikasi sentimen Bahasa Indonesia hasil fine-tuning.

- **Model dasar:** IndoBERT
- **Kelas:** {kelas}
- **Dilatih dengan:** {sumber_data}

## Cara pakai

```python
from transformers import pipeline
pipe = pipeline("sentiment-analysis", model="{repo}")
pipe("pelayanannya sangat memuaskan")
```

## Catatan

Model ini dilatih untuk pemantauan opini publik berbahasa Indonesia.
Performa di luar ranah data latihnya bisa menurun. Selalu evaluasi ulang
pada data milikmu sendiri sebelum dipakai untuk pengambilan keputusan.
"""


def main():
    ap = argparse.ArgumentParser(description="Unggah model ke HuggingFace Hub.")
    ap.add_argument("--repo", required=True, help="tujuan, mis. namamu/nama-model")
    ap.add_argument("--dir", default=DEFAULT_LOCAL, help="folder model lokal")
    ap.add_argument("--public", action="store_true",
                    help="jadikan repo PUBLIK (bawaan: privat)")
    ap.add_argument("--yes", action="store_true",
                    help="konfirmasi unggah; tanpa ini hanya pratinjau")
    ap.add_argument("--sumber-data", default="data internal + dataset berlabel HuggingFace")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: folder model tidak ditemukan: {args.dir}")
        print("Latih dulu: python -m analysis.finetune_indobert --csv data/latih.csv")
        return 1

    token = os.environ.get("HF_TOKEN", "")
    berkas = sorted(os.listdir(args.dir))
    ukuran = sum(os.path.getsize(os.path.join(args.dir, f))
                 for f in berkas if os.path.isfile(os.path.join(args.dir, f)))

    print("── Rencana unggah ──")
    print(f"  Dari     : {args.dir}")
    print(f"  Ke       : {args.repo}")
    print(f"  Visibilitas: {'PUBLIK ⚠️' if args.public else 'privat'}")
    print(f"  Ukuran   : {ukuran / 1024 / 1024:.0f} MB, {len(berkas)} berkas")
    print(f"  Token    : {'ditemukan' if token else 'TIDAK ADA (set HF_TOKEN dulu)'}")

    if not args.yes:
        print("\nIni baru PRATINJAU — belum ada yang dikirim.")
        print("Tambahkan --yes bila benar ingin mengunggah.")
        return 0
    if not token:
        print("\nBatal: HF_TOKEN belum diset.")
        return 1
    if args.public:
        print("\n⚠️  Repo akan PUBLIK — siapa pun bisa mengunduh model ini.")
        jawab = input("Ketik 'PUBLIK' untuk melanjutkan: ").strip()
        if jawab != "PUBLIK":
            print("Dibatalkan.")
            return 1

    try:
        from huggingface_hub import HfApi
    except Exception:
        print("ERROR: huggingface_hub belum terpasang.")
        return 1

    # tulis kartu model bila belum ada
    kartu = os.path.join(args.dir, "README.md")
    if not os.path.isfile(kartu):
        kelas = "positive / neutral / negative"
        try:
            import json
            with open(os.path.join(args.dir, "config.json"), encoding="utf-8") as f:
                id2 = json.load(f).get("id2label", {})
            if id2:
                kelas = " / ".join(str(v) for v in id2.values())
        except Exception:
            pass
        with open(kartu, "w", encoding="utf-8") as f:
            f.write(KARTU_MODEL.format(repo=args.repo, kelas=kelas,
                                       sumber_data=args.sumber_data))
        print("Kartu model (README.md) dibuat.")

    api = HfApi(token=token)
    print("\nMengunggah...")
    api.create_repo(repo_id=args.repo, private=not args.public, exist_ok=True)
    api.upload_folder(folder_path=args.dir, repo_id=args.repo)
    print(f"Selesai: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
