"""Membangun notebooks/finetune_colab.ipynb dari sel-sel di bawah.

Notebook dibuat lewat skrip (bukan diedit manual) supaya JSON-nya selalu valid
dan logikanya tetap sama dengan analysis/finetune_indobert.py.

Jalankan:  python tools/buat_notebook_colab.py [baseline_f1_sarkasme]
"""
from __future__ import annotations

import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebooks", "finetune_colab.ipynb")
BASELINE = sys.argv[1] if len(sys.argv) > 1 else "belum diukur"


def md(teks):
    return {"cell_type": "markdown", "metadata": {}, "source": teks.strip("\n")}


def code(teks):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": teks.strip("\n")}


SEL = [
md(f"""
# Fine-tuning Bahasa Indonesia di Colab (GPU gratis) — Sentimen & Sarkasme

Satu notebook untuk dua tugas. Pilih `TUGAS` di sel **Konfigurasi**, lalu
`Runtime > Run all`.

| Tugas | Data (berlabel manusia) | Model dasar | Kelas |
|---|---|---|---|
| `sentimen` | carant-ai (21.000 seimbang) | `w11wo/indonesian-roberta-base-sentiment-classifier` | positive / neutral / negative |
| `sarkasme` | `w11wo/twitter_indonesia_sarcastic` — split resmi | `indolem/indobertweet-base-uncased` | bukan_sarkas / sarkas |

**Protokol (menghindari kesalahan umum):**
1. **Urutan label ikut model dasar.** Kalau model dasar sudah punya label yang sama,
   urutannya dipertahankan — kalau diacak, pengetahuan model terbuang.
2. **Bobot kelas** untuk data tidak seimbang (sarkasme ≈ 1 : 3).
3. **Latih / validasi / uji terpisah.** Epoch terbaik dipilih dari validasi;
   angka akhir dari data **uji** yang tak pernah dilihat. Tidak ada SMOTE atau
   TF-IDF yang di-fit ke seluruh data (sumber kebocoran).

Pembanding sarkasme: model jadi `w11wo/indobert-base-p1-twitter-indonesia-sarcastic`
mencapai **F1 (kelas sarkas) {BASELINE}** pada split uji resmi. Hasilmu layak
dipakai bila menyamai atau melampaui angka itu.

**Sebelum mulai:** `Runtime > Change runtime type > GPU`.
"""),
code("""
# 1) Pasang dependensi
!pip -q install -U transformers datasets scikit-learn accelerate sentencepiece
"""),
code("""
# 2) Pastikan GPU aktif
import torch
print('GPU tersedia :', torch.cuda.is_available())
print('Perangkat    :', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'CPU — AKTIFKAN GPU: Runtime > Change runtime type > GPU')
"""),
md("## 3) Konfigurasi"),
code("""
TUGAS = "sentimen"        # "sentimen" atau "sarkasme"
SUMBER_DATA = "hf"        # "hf" = unduh dari HuggingFace | "upload" = CSV dari laptop
JUMLAH_SENTIMEN = 21000   # total contoh sentimen (dibagi rata 3 kelas)

KONFIG = {
    "sentimen": dict(base="w11wo/indonesian-roberta-base-sentiment-classifier",
                     epochs=3, batch=32, lr=2e-5, maxlen=128,
                     out="indobert-sentiment-finetuned"),
    "sarkasme": dict(base="indolem/indobertweet-base-uncased",
                     epochs=5, batch=32, lr=3e-5, maxlen=128,
                     out="sarkasme-finetuned"),
}
C = KONFIG[TUGAS]
print(TUGAS, C)
"""),
md("""
## 4) Data

- `SUMBER_DATA = "hf"` — diunduh langsung, tak perlu upload.
- `SUMBER_DATA = "upload"` — pilih CSV dari laptop, mis. `data/latih_20k.csv`
  (sentimen) atau `data/latih_sarkasme.csv` (sarkasme, berkolom `split`).
"""),
code("""
import csv, io, random
from collections import Counter
from itertools import islice

def dari_csv(isi):
    d = {}
    for row in csv.DictReader(io.StringIO(isi)):
        t = (row.get('text') or row.get('tweet') or '').strip()
        l = (row.get('label') or '').strip().lower()
        s = (row.get('split') or 'semua').strip().lower()
        if t and l:
            d.setdefault(s, ([], []))
            d[s][0].append(t); d[s][1].append(l)
    return d

if SUMBER_DATA == 'upload':
    from google.colab import files
    up = files.upload()
    data = dari_csv(up[list(up)[0]].decode('utf-8'))

elif TUGAS == 'sarkasme':
    from datasets import load_dataset
    PETA = {0: 'bukan_sarkas', 1: 'sarkas'}   # kartu dataset: 0 = non-sarcastic, 1 = sarcastic
    data = {}
    for s in ['train', 'validation', 'test']:          # split RESMI dipertahankan
        ds = load_dataset('w11wo/twitter_indonesia_sarcastic', split=s)
        data[s] = ([r['tweet'] for r in ds], [PETA[int(r['label'])] for r in ds])

else:
    from datasets import load_dataset
    ds = load_dataset('carant-ai/indonesian_sentiment_dataset', split='train', streaming=True)
    per = {}
    for row in islice(ds, JUMLAH_SENTIMEN * 4):
        t = (row.get('text') or '').strip()
        l = str(row.get('label_text') or '').lower()
        if t and l in ('positive', 'neutral', 'negative'):
            per.setdefault(l, []).append(t)
    n = min(min(len(v) for v in per.values()), JUMLAH_SENTIMEN // 3)
    pasangan = [(t, l) for l, v in per.items() for t in v[:n]]
    random.Random(42).shuffle(pasangan)   # WAJIB diacak — tanpa ini data terurut per kelas
    data = {'semua': ([t for t, _ in pasangan], [l for _, l in pasangan])}

for s, (t, l) in data.items():
    print(f'{s:<10} {len(t):>6}  {dict(Counter(l))}')
"""),
code("""
# 5) Latih / validasi / uji — split resmi dipakai bila ada
from sklearn.model_selection import train_test_split

def strat(t, l, porsi):
    a, b, c, d = train_test_split(t, l, test_size=porsi, random_state=42, stratify=l)
    return (a, c), (b, d)

if {'train', 'test'} <= set(data):
    if 'validation' not in data:
        data['train'], data['validation'] = strat(*data['train'], 0.1)
else:
    t = sum((v[0] for v in data.values()), [])
    l = sum((v[1] for v in data.values()), [])
    latih, sisa = strat(t, l, 0.2)
    val, uji = strat(*sisa, 0.5)
    data = {'train': latih, 'validation': val, 'test': uji}

for s in ['train', 'validation', 'test']:
    print(f'{s:<10} {len(data[s][0]):>6}  {dict(Counter(data[s][1]))}')
"""),
md("""
## 6) Selaraskan urutan label dengan model dasar

Setiap model menyimpan urutan labelnya sendiri, misalnya
`w11wo`: 0=positive, 1=neutral, 2=negative — sedangkan IndoBERTweet-sentimen justru
0=Negative. Sel ini membaca urutan dari config model dasar dan memakainya bila
labelnya cocok dengan data. Bila tidak cocok (mis. model dasar polos), dibuat
kepala klasifikasi baru.
"""),
code("""
from transformers import AutoConfig

ALIAS = {'positive': 'positive', 'positif': 'positive', 'neutral': 'neutral',
         'netral': 'neutral', 'negative': 'negative', 'negatif': 'negative',
         'sarkas': 'sarkas', 'sarcastic': 'sarkas', 'bukan_sarkas': 'bukan_sarkas',
         'non_sarcastic': 'bukan_sarkas'}
# Model yang config-nya hanya LABEL_0/1/2 — artinya harus dari kartu model, bukan ditebak.
OVERRIDE = {'mdhugol/indonesia-bert-sentiment-classification':
            {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}}

kelas = sorted(set(data['train'][1]))
id2 = getattr(AutoConfig.from_pretrained(C['base']), 'id2label', {}) or {}
ov = OVERRIDE.get(C['base'], {})
nama = {int(i): ov.get(v) or ALIAS.get(str(v).lower()) for i, v in id2.items()}

if len(nama) == len(kelas) and set(nama.values()) == set(kelas):
    label2id = {lab: i for i, lab in nama.items()}
    print('✅ Urutan label IKUT model dasar:', dict(sorted(nama.items())))
else:
    label2id = {l: i for i, l in enumerate(kelas)}
    print('ℹ️  Kepala klasifikasi BARU:', label2id, '| label model dasar:', dict(id2))
id2label = {i: l for l, i in label2id.items()}
"""),
code("""
# 7) Latih
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score

torch.manual_seed(42); np.random.seed(42); random.seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(C['base'])
model = AutoModelForSequenceClassification.from_pretrained(
    C['base'], num_labels=len(kelas), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True).to(dev)

def loader(split, acak):
    t, l = data[split]
    e = tok(t, truncation=True, padding='max_length', max_length=C['maxlen'], return_tensors='pt')
    y = torch.tensor([label2id[x] for x in l])
    return DataLoader(TensorDataset(e['input_ids'], e['attention_mask'], y),
                      batch_size=C['batch'], shuffle=acak)

dl_latih, dl_val, dl_uji = loader('train', True), loader('validation', False), loader('test', False)

# bobot kelas N / (K * n_c) — kelas langka diberi bobot lebih besar
hit = Counter(label2id[x] for x in data['train'][1]); N, K = sum(hit.values()), len(kelas)
bobot = torch.tensor([N / (K * hit[i]) for i in range(K)], dtype=torch.float, device=dev)
print('Bobot kelas:', {id2label[i]: round(float(w), 3) for i, w in enumerate(bobot)})
kriteria = torch.nn.CrossEntropyLoss(weight=bobot)

opt = torch.optim.AdamW(model.parameters(), lr=C['lr'], weight_decay=0.01)
total = len(dl_latih) * C['epochs']
jadwal = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
amp = dev.type == 'cuda'
scaler = torch.amp.GradScaler('cuda') if amp else None

def prediksi(dl):
    model.eval(); p, a = [], []
    with torch.no_grad():
        for ids, mask, y in dl:
            with torch.autocast(dev.type, enabled=amp):
                lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            p += lg.argmax(-1).cpu().tolist(); a += y.tolist()
    return a, p

terbaik, state = -1, None
for ep in range(1, C['epochs'] + 1):
    model.train(); rugi = 0
    for ids, mask, y in dl_latih:
        ids, mask, y = ids.to(dev), mask.to(dev), y.to(dev)
        opt.zero_grad()
        with torch.autocast(dev.type, enabled=amp):
            loss = kriteria(model(input_ids=ids, attention_mask=mask).logits, y)
        if scaler: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else: loss.backward(); opt.step()
        jadwal.step(); rugi += loss.item()
    a, p = prediksi(dl_val)
    f1 = f1_score(a, p, average='macro', zero_division=0)
    tanda = ''
    if f1 > terbaik:
        terbaik, tanda = f1, '  ← terbaik'
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'epoch {ep}: loss {rugi / len(dl_latih):.4f} | F1 validasi {f1:.4f}{tanda}')

model.load_state_dict(state)
"""),
code("""
# 8) Nilai akhir pada data UJI (tak pernah dilihat saat latihan)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

PEMBANDING_SARKASME = {BASELINE!r}   # F1 kelas sarkas model jadi w11wo (split uji resmi)

a, p = prediksi(dl_uji)
nama_kelas = [id2label[i] for i in range(K)]
print(classification_report(a, p, target_names=nama_kelas, digits=4, zero_division=0))
if TUGAS == 'sarkasme':
    f1_sarkas = f1_score(a, p, pos_label=label2id['sarkas'], average='binary')
    print(f'F1 kelas sarkas: {f1_sarkas:.4f}   (pembanding model jadi: {PEMBANDING_SARKASME})')
ConfusionMatrixDisplay(confusion_matrix(a, p), display_labels=nama_kelas).plot(cmap='Blues')
plt.title(f'Confusion matrix — {TUGAS} (data uji)'); plt.show()
""".replace("{BASELINE!r}", repr(BASELINE))),
code("""
# 9) Simpan & unduh
model.save_pretrained(C['out']); tok.save_pretrained(C['out'])
!zip -qr {C['out']}.zip {C['out']}
from google.colab import files
files.download(f"{C['out']}.zip")
print(f"Ekstrak ke folder proyek: models/{C['out']}/")
if TUGAS == 'sentimen':
    print('lalu set config.yaml -> sentiment.model_dir: "models/indobert-sentiment-finetuned"')
else:
    print('analysis/sarcasm.py otomatis memakai folder itu bila ada.')
"""),
code("""
# 10) (Opsional) Unggah ke HuggingFace Hub — REPO PRIVAT
# Token: https://huggingface.co/settings/tokens (akses Write). Jangan dibagikan.
# from huggingface_hub import HfApi
# TOKEN = ''
# REPO = 'namamu/' + C['out']
# api = HfApi(token=TOKEN)
# api.create_repo(REPO, private=True, exist_ok=True)
# api.upload_folder(folder_path=C['out'], repo_id=REPO)
# print('https://huggingface.co/' + REPO)
"""),
]


def _baris(src: str) -> list:
    lines = src.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main():
    for c in SEL:
        c["source"] = _baris(c["source"])
    nb = {"cells": SEL,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Notebook ditulis: {OUT} ({len(SEL)} sel)")


if __name__ == "__main__":
    main()
