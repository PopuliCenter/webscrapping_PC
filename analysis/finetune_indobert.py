"""Fine-tuning IndoBERT dengan data berlabel sendiri.

BEDA dengan Sastrawi: menambah kata ke kamus Sastrawi berlaku seketika,
sedangkan IndoBERT harus DILATIH ULANG dengan CONTOH KALIMAT berlabel —
bukan sekadar daftar kata. Tokenizer BERT memecah kata asing jadi sub-kata,
jadi kata baru tetap terbaca; yang perlu diajarkan adalah MAKNANYA dalam
konteks kalimat.

Sumber data latih:
  - CSV berkolom `text,label`  (label: positive / neutral / negative)
  - atau langsung dari database monitoring

Jalankan:
    python -m analysis.finetune_indobert --csv data/latih.csv --epochs 3
    python -m analysis.finetune_indobert --from-db --epochs 3

Hasil disimpan ke models/indobert-sentiment-finetuned/ (tidak menimpa model
asli). Arahkan config.yaml -> sentiment.model_dir ke folder itu untuk memakainya.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.sentiment import HUB_MODEL, DEFAULT_MODEL_DIR, _is_local_model  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "indobert-sentiment-finetuned")


# ── Data ────────────────────────────────────────────────────────
def load_csv(path: str) -> tuple:
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or row.get("teks") or "").strip()
            l = (row.get("label") or row.get("sentimen") or "").strip().lower()
            if t and l:
                texts.append(t)
                labels.append(l)
    return texts, labels


def load_db(db_path: str) -> tuple:
    from analysis.ml_classify import load_dataset_from_db
    return load_dataset_from_db(db_path)


# ── Pelatihan ───────────────────────────────────────────────────
def finetune(texts: list, labels: list, base_model: str = "", epochs: int = 3,
             batch_size: int = 16, lr: float = 2e-5, max_len: int = 128,
             test_size: float = 0.2, output_dir: str = "", seed: int = 42) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        return {"error": f"Butuh transformers + torch: {e}"}

    if len(texts) < 30:
        return {"error": f"Data terlalu sedikit ({len(texts)}). Minimal ~30 contoh, "
                         f"idealnya ratusan per kelas."}

    kelas = sorted(set(labels))
    if len(kelas) < 2:
        return {"error": "Butuh minimal 2 kelas label."}

    label2id = {l: i for i, l in enumerate(kelas)}
    id2label = {i: l for l, i in label2id.items()}
    base = base_model or (DEFAULT_MODEL_DIR if _is_local_model(DEFAULT_MODEL_DIR) else HUB_MODEL)
    output_dir = output_dir or OUTPUT_DIR

    random.seed(seed)
    torch.manual_seed(seed)

    # acak & bagi data
    data = list(zip(texts, [label2id[l] for l in labels]))
    random.shuffle(data)
    n_test = max(1, int(len(data) * test_size))
    test_data, train_data = data[:n_test], data[n_test:]

    print(f"Model dasar : {base}")
    print(f"Kelas       : {kelas}")
    print(f"Data latih  : {len(train_data)} | uji: {len(test_data)}")
    print(f"Distribusi  : {Counter(labels)}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(kelas), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Perangkat   : {device}")

    def to_ds(pairs):
        txt = [t for t, _ in pairs]
        y = torch.tensor([l for _, l in pairs])
        enc = tok(txt, truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_dl = DataLoader(to_ds(train_data), batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(to_ds(test_data), batch_size=batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    riwayat = []

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            opt.step()
            total += out.loss.item()
        rata_loss = total / max(len(train_dl), 1)

        # evaluasi
        model.eval()
        benar = jml = 0
        with torch.no_grad():
            for ids, mask, y in test_dl:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                pred = model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1)
                benar += (pred == y).sum().item()
                jml += y.size(0)
        akurasi = benar / max(jml, 1)
        riwayat.append({"epoch": ep, "loss": round(rata_loss, 4),
                        "akurasi_uji": round(akurasi, 4)})
        print(f"  epoch {ep}/{epochs} — loss {rata_loss:.4f} | akurasi uji {akurasi:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"\nModel tersimpan di: {output_dir}")
    print("Pakai dengan menambah di config.yaml:")
    print(f'  sentiment:\n    model_dir: "models/indobert-sentiment-finetuned"')

    return {"output_dir": output_dir, "kelas": kelas, "riwayat": riwayat,
            "akurasi_akhir": riwayat[-1]["akurasi_uji"] if riwayat else None}


def main():
    ap = argparse.ArgumentParser(description="Fine-tuning IndoBERT untuk sentimen.")
    ap.add_argument("--csv", help="CSV berkolom text,label")
    ap.add_argument("--from-db", action="store_true", help="ambil data dari database")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--output", default="", help="folder keluaran model")
    args = ap.parse_args()

    if args.csv:
        texts, labels = load_csv(args.csv)
    elif args.from_db:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"), encoding="utf-8"))
        texts, labels = load_db(os.path.join(PROJECT_DIR, cfg["storage"]["db_path"]))
        print("CATATAN: label dari database berasal dari IndoBERT itu sendiri, "
              "sehingga pelatihan menjadi sirkular. Untuk hasil sahih gunakan "
              "data berlabel manual lewat --csv.")
    else:
        ap.error("Pilih salah satu: --csv <file> atau --from-db")

    if not texts:
        print("Tidak ada data latih ditemukan.")
        return

    res = finetune(texts, labels, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, output_dir=args.output)
    if "error" in res:
        print("Error:", res["error"])


if __name__ == "__main__":
    main()
