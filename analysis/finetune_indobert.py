"""Fine-tuning model klasifikasi teks (sentimen / sarkasme) dengan data berlabel.

Menjalankan di laptop (CPU) bisa, tapi lambat. Untuk data ribuan baris pakai
notebooks/finetune_colab.ipynb (GPU gratis) — logikanya sama dengan berkas ini.

Protokol yang dipakai (menghindari kesalahan umum):
  1. URUTAN LABEL IKUT MODEL DASAR. Bila model dasar sudah punya kepala
     klasifikasi dengan label yang sama (mis. w11wo: positive/neutral/negative),
     urutannya dipertahankan — kalau diacak, "pengetahuan" model terbuang.
  2. BOBOT KELAS untuk data tidak seimbang (mis. sarkasme ~1:3).
  3. TIGA SPLIT: latih / validasi / uji. Epoch terbaik dipilih dari validasi,
     angka akhir dilaporkan dari data UJI yang tak pernah dilihat.
     Bila CSV punya kolom `split`, split resmi itu yang dipakai.

Jalankan:
    python -m analysis.finetune_indobert --tugas sentimen --csv data/latih_20k.csv
    python -m analysis.finetune_indobert --tugas sarkasme --from-hf
    python -m analysis.finetune_indobert --tugas sentimen --csv data/latih.csv --base <model>
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.sentiment import HUB_MODEL, DEFAULT_MODEL_DIR, _is_local_model  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TUGAS = {
    "sentimen": {
        # pemenang uji banding (acc 0.610 vs 0.548 mdhugol)
        "base": "w11wo/indonesian-roberta-base-sentiment-classifier",
        "output": os.path.join(PROJECT_DIR, "models", "indobert-sentiment-finetuned"),
    },
    "sarkasme": {
        # IndoBERTweet: dilatih dari tweet -> cocok untuk data X
        "base": "indolem/indobertweet-base-uncased",
        "output": os.path.join(PROJECT_DIR, "models", "sarkasme-finetuned"),
    },
}

# Pemetaan label mentah -> nama kelas. Hanya nama yang JELAS yang dipetakan.
_ALIAS = {
    "positive": "positive", "positif": "positive", "pos": "positive",
    "neutral": "neutral", "netral": "neutral", "net": "neutral",
    "negative": "negative", "negatif": "negative", "neg": "negative",
    "sarkas": "sarkas", "sarcastic": "sarkas",
    "bukan_sarkas": "bukan_sarkas", "non_sarcastic": "bukan_sarkas",
    "not_sarcastic": "bukan_sarkas",
}


# ── Data ────────────────────────────────────────────────────────
def load_csv(path: str) -> dict:
    """-> {split: (teks, label)}. Tanpa kolom `split`, semua masuk 'semua'."""
    data = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or row.get("teks") or row.get("tweet") or "").strip()
            l = (row.get("label") or row.get("sentimen") or "").strip().lower()
            s = (row.get("split") or "semua").strip().lower()
            if t and l:
                data.setdefault(s, ([], []))
                data[s][0].append(t)
                data[s][1].append(l)
    return data


def bagi_data(data: dict, seed: int = 42) -> dict:
    """Pastikan ada train/validation/test. Split resmi dipakai bila ada."""
    if {"train", "test"} <= set(data):
        if "validation" not in data:            # ambil 10% latih jadi validasi
            data["train"], data["validation"] = _stratified(*data["train"], 0.1, seed)
        return data
    teks, label = [], []
    for t, l in data.values():
        teks += t
        label += l
    latih, sisa = _stratified(teks, label, 0.2, seed)
    val, uji = _stratified(*sisa, 0.5, seed)
    return {"train": latih, "validation": val, "test": uji}


def _stratified(teks, label, porsi, seed):
    from sklearn.model_selection import train_test_split
    a, b, c, d = train_test_split(teks, label, test_size=porsi,
                                  random_state=seed, stratify=label)
    return (a, c), (b, d)


# ── Penyelarasan label ──────────────────────────────────────────
def _override_untuk(base: str) -> dict:
    """Override untuk model yang config-nya hanya LABEL_0/1/2 (tak informatif)."""
    from analysis.hf_models import MODEL_REGISTRY
    if os.path.abspath(base) == os.path.abspath(DEFAULT_MODEL_DIR) or base == HUB_MODEL:
        return MODEL_REGISTRY["mdhugol"]["label_override"]
    for cfg in MODEL_REGISTRY.values():
        if cfg["id"] == base and cfg.get("label_override"):
            return cfg["label_override"]
    return {}


def selaraskan_label(base: str, kelas: list) -> tuple:
    """-> (label2id, keterangan). Ikuti urutan model dasar bila labelnya cocok."""
    from transformers import AutoConfig
    try:
        id2 = getattr(AutoConfig.from_pretrained(base), "id2label", {}) or {}
    except Exception:
        id2 = {}
    override = _override_untuk(base)
    nama = {}
    for i, v in id2.items():
        n = override.get(v) or _ALIAS.get(str(v).strip().lower())
        nama[int(i)] = n
    if len(nama) == len(kelas) and set(nama.values()) == set(kelas):
        return ({lab: i for i, lab in nama.items()},
                f"IKUT model dasar {dict(sorted(nama.items()))} — kepala klasifikasi dipakai ulang")
    urut = sorted(kelas)
    return ({l: i for i, l in enumerate(urut)},
            f"kepala klasifikasi BARU {dict(enumerate(urut))} (label model dasar: "
            f"{dict(id2) or 'tidak ada'})")


# ── Pelatihan ───────────────────────────────────────────────────
def finetune(data: dict, base: str, output_dir: str, epochs: int = 3,
             batch_size: int = 16, lr: float = 2e-5, max_len: int = 128,
             seed: int = 42) -> dict:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  get_linear_schedule_with_warmup)
        from sklearn.metrics import f1_score, accuracy_score, classification_report
    except Exception as e:
        return {"error": f"Butuh transformers + torch + scikit-learn: {e}"}

    data = bagi_data(data, seed)
    kelas = sorted(set(data["train"][1]))
    if len(kelas) < 2:
        return {"error": "Butuh minimal 2 kelas label."}
    if len(data["train"][0]) < 30:
        return {"error": f"Data latih terlalu sedikit ({len(data['train'][0])})."}

    label2id, ket = selaraskan_label(base, kelas)
    id2label = {i: l for l, i in label2id.items()}
    for s in ("train", "validation", "test"):
        print(f"  {s:<10}: {len(data[s][0]):>6}  {dict(Counter(data[s][1]))}")
    print(f"Model dasar : {base}")
    print(f"Label       : {ket}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Perangkat   : {device}")

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(kelas), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)

    def dl(split, acak):
        teks, lab = data[split]
        enc = tok(teks, truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        y = torch.tensor([label2id[l] for l in lab])
        return DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y),
                          batch_size=batch_size, shuffle=acak)

    dl_latih, dl_val, dl_uji = dl("train", True), dl("validation", False), dl("test", False)

    # bobot kelas: N / (K * n_c) — kelas langka diberi bobot lebih besar
    hitung = Counter(label2id[l] for l in data["train"][1])
    n, k = sum(hitung.values()), len(kelas)
    bobot = torch.tensor([n / (k * hitung[i]) for i in range(k)],
                         dtype=torch.float, device=device)
    print(f"Bobot kelas : { {id2label[i]: round(float(w), 3) for i, w in enumerate(bobot)} }")
    kriteria = torch.nn.CrossEntropyLoss(weight=bobot)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_langkah = len(dl_latih) * epochs
    jadwal = get_linear_schedule_with_warmup(opt, int(0.1 * total_langkah), total_langkah)
    pakai_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if pakai_amp else None

    def prediksi(loader):
        model.eval()
        pred, asli = [], []
        with torch.no_grad():
            for ids, mask, y in loader:
                with torch.autocast(device.type, enabled=pakai_amp):
                    logit = model(input_ids=ids.to(device),
                                  attention_mask=mask.to(device)).logits
                pred += logit.argmax(-1).cpu().tolist()
                asli += y.tolist()
        return asli, pred

    terbaik, state_terbaik, riwayat = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in dl_latih:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=pakai_amp):
                loss = kriteria(model(input_ids=ids, attention_mask=mask).logits, y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            jadwal.step()
            total += loss.item()
        y, p = prediksi(dl_val)
        f1 = f1_score(y, p, average="macro", zero_division=0)
        riwayat.append({"epoch": ep, "loss": round(total / len(dl_latih), 4),
                        "f1_validasi": round(f1, 4)})
        tanda = ""
        if f1 > terbaik:
            terbaik, tanda = f1, "  ← terbaik"
            state_terbaik = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
        print(f"  epoch {ep}/{epochs} — loss {total / len(dl_latih):.4f} | "
              f"F1 validasi {f1:.4f}{tanda}")

    model.load_state_dict(state_terbaik)
    y, p = prediksi(dl_uji)
    nama_kelas = [id2label[i] for i in range(k)]
    print("\n=== Hasil pada data UJI (tidak pernah dilihat saat latihan) ===")
    print(classification_report(y, p, target_names=nama_kelas, digits=4, zero_division=0))

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Model tersimpan di: {output_dir}")

    return {"output_dir": output_dir, "label2id": label2id, "riwayat": riwayat,
            "akurasi_uji": round(accuracy_score(y, p), 4),
            "f1_uji": round(f1_score(y, p, average="macro", zero_division=0), 4)}


def main():
    ap = argparse.ArgumentParser(description="Fine-tuning klasifikasi teks Indonesia.")
    ap.add_argument("--tugas", choices=list(TUGAS), default="sentimen")
    ap.add_argument("--csv", help="CSV berkolom text,label (opsional: split)")
    ap.add_argument("--from-hf", action="store_true",
                    help="ambil data berlabel manusia langsung dari HuggingFace")
    ap.add_argument("--from-db", action="store_true",
                    help="(sentimen) ambil dari database — SIRKULAR, hanya untuk uji")
    ap.add_argument("--hf-limit", type=int, default=21000,
                    help="(sentimen --from-hf) jumlah contoh seimbang")
    ap.add_argument("--base", default="", help="model dasar (default per tugas)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    cfg = TUGAS[args.tugas]
    if args.csv:
        data = load_csv(args.csv)
    elif args.from_hf:
        from analysis.hf_datasets import load_labeled, load_task
        if args.tugas == "sarkasme":
            data = load_task("sarkasme")
        else:
            t, l = load_labeled("carant", limit=args.hf_limit * 4, seimbang=True)
            data = {"semua": (t[:args.hf_limit], l[:args.hf_limit])}
    elif args.from_db and args.tugas == "sentimen":
        import yaml
        from analysis.ml_classify import load_dataset_from_db
        c = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"), encoding="utf-8"))
        t, l = load_dataset_from_db(os.path.join(PROJECT_DIR, c["storage"]["db_path"]))
        print("CATATAN: label dari database dihasilkan model itu sendiri — pelatihan "
              "SIRKULAR. Pakai --csv / --from-hf untuk hasil yang sahih.")
        data = {"semua": (t, l)}
    else:
        ap.error("Pilih sumber data: --csv <file>, --from-hf, atau --from-db")

    if not any(v[0] for v in data.values()):
        print("Tidak ada data latih ditemukan.")
        return
    res = finetune(data, base=args.base or cfg["base"], output_dir=args.output or cfg["output"],
                   epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   max_len=args.max_len)
    if "error" in res:
        print("Error:", res["error"])
    elif args.tugas == "sentimen":
        print('\nPakai dengan: config.yaml -> sentiment.model_dir: '
              '"models/indobert-sentiment-finetuned"')
    else:
        print("\nModel sarkasme otomatis dipakai analysis/sarcasm.py bila folder ini ada.")


if __name__ == "__main__":
    main()
