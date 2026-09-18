"""Deteksi SARKASME pada teks Bahasa Indonesia.

Urutan model yang dipakai (yang pertama tersedia):
  1. models/sarkasme-finetuned/   — hasil fine-tuning SENDIRI (notebook Colab)
  2. models/sarkasme-w11wo/       — model jadi, disimpan lokal
  3. w11wo/indobert-base-p1-twitter-indonesia-sarcastic (unduh dari Hub)

Model jadi w11wo hanya menyebut label `LABEL_0/LABEL_1` di config-nya. Artinya
TIDAK ditebak: dipastikan secara empiris dengan menguji kedua kemungkinan
pemetaan pada split uji resmi (lihat `verifikasi_label()`), dan pemetaan yang
dipakai dicatat di PEMETAAN_TERVERIFIKASI.

Pakai:
    from analysis.sarcasm import SarcasmEngine
    SarcasmEngine().predict(["wah hebat sekali, jalan rusak dibiarkan 3 tahun"])
"""
from __future__ import annotations

import os
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_FINETUNED = os.path.join(PROJECT_DIR, "models", "sarkasme-finetuned")
DIR_W11WO = os.path.join(PROJECT_DIR, "models", "sarkasme-w11wo")
HUB_MODEL = "w11wo/indobert-base-p1-twitter-indonesia-sarcastic"
AMBANG_FILE = os.path.join(PROJECT_DIR, "resources", "ambang_sarkasme.json")

# TERVERIFIKASI oleh verifikasi_label() pada 538 tweet uji resmi:
#   LABEL_1 = sarkas -> akurasi 0.8662, F1 sarkas 0.7273  (sama persis dgn angka resmi)
#   LABEL_0 = sarkas -> akurasi 0.1338, F1 sarkas 0.1402
PEMETAAN_TERVERIFIKASI = {"LABEL_0": "bukan_sarkas", "LABEL_1": "sarkas"}

_NAMA = {"sarkas": "sarkas", "sarcastic": "sarkas",
         "bukan_sarkas": "bukan_sarkas", "non_sarcastic": "bukan_sarkas",
         "not_sarcastic": "bukan_sarkas"}


def _model_lokal(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "config.json")) and
            any(os.path.isfile(os.path.join(path, f))
                for f in ("model.safetensors", "pytorch_model.bin")))


def pilih_sumber() -> str:
    for p in (DIR_FINETUNED, DIR_W11WO):
        if _model_lokal(p):
            return p
    return HUB_MODEL


class SarcasmEngine:
    def __init__(self, sumber: str = ""):
        from transformers import pipeline, AutoConfig
        self.sumber = sumber or pilih_sumber()
        id2 = getattr(AutoConfig.from_pretrained(self.sumber), "id2label", {}) or {}
        self._peta = {}
        for v in id2.values():
            nama = _NAMA.get(str(v).lower()) or PEMETAAN_TERVERIFIKASI.get(v)
            if not nama:
                raise ValueError(f"Label model sarkasme tak dikenal: {v}. "
                                 f"Tambahkan pemetaannya setelah diverifikasi.")
            self._peta[v] = nama
        self._pipe = pipeline("text-classification", model=self.sumber,
                              tokenizer=self.sumber, top_k=None)
        self.ambang = _muat_ambang()

    def predict(self, texts: list, batch_size: int = 32) -> list:
        """-> [(label, peluang_sarkas), ...]"""
        out = self._pipe([(t or "")[:512] for t in texts],
                         batch_size=batch_size, truncation=True)
        hasil = []
        for per_teks in out:
            skor = {self._peta[o["label"]]: float(o["score"]) for o in per_teks}
            p = skor.get("sarkas", 0.0)
            hasil.append(("sarkas" if p >= self.ambang else "bukan_sarkas", round(p, 4)))
        return hasil


def _muat_ambang() -> float:
    """Ambang peluang untuk menyebut 'sarkas'. Bawaan 0.5."""
    import json
    if os.path.isfile(AMBANG_FILE):
        with open(AMBANG_FILE, encoding="utf-8") as f:
            return float(json.load(f).get("ambang", 0.5))
    return 0.5


def _data_split(split: str) -> tuple:
    """(teks, label 0/1) satu split dari data/latih_sarkasme.csv."""
    import csv
    teks, y = [], []
    with open(os.path.join(PROJECT_DIR, "data", "latih_sarkasme.csv"),
              encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                teks.append(row["text"])
                y.append(1 if row["label"] == "sarkas" else 0)
    return teks, y


def pilih_ambang(presisi_min: float = 0.85, simpan: bool = True) -> dict:
    """Pilih ambang di data VALIDASI, laporkan di data UJI (tanpa kebocoran).

    Dua pilihan:
      - "seimbang"      : F1 tertinggi di validasi
      - "presisi_tinggi": ambang terkecil dgn presisi >= presisi_min di validasi
                          (lebih sedikit keluhan tulus yang salah dicap sarkas)
    """
    import json
    from sklearn.metrics import precision_score, recall_score, f1_score

    eng = SarcasmEngine()
    tv, yv = _data_split("validation")
    tu, yu = _data_split("test")
    pv = [p for _, p in eng.predict(tv)]
    pu = [p for _, p in eng.predict(tu)]

    def ukur(y, p, a):
        pred = [1 if x >= a else 0 for x in p]
        return {"presisi": round(precision_score(y, pred, zero_division=0), 4),
                "recall": round(recall_score(y, pred, zero_division=0), 4),
                "f1": round(f1_score(y, pred, zero_division=0), 4)}

    kandidat = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    tabel = {a: {"validasi": ukur(yv, pv, a), "uji": ukur(yu, pu, a)} for a in kandidat}
    seimbang = max(kandidat, key=lambda a: tabel[a]["validasi"]["f1"])
    lolos = [a for a in kandidat if tabel[a]["validasi"]["presisi"] >= presisi_min]
    presisi_tinggi = min(lolos) if lolos else None

    print(f"{'ambang':>7} | {'VALIDASI  presisi recall  f1':<30} | UJI  presisi recall  f1")
    for a in kandidat:
        v, u = tabel[a]["validasi"], tabel[a]["uji"]
        tanda = " ← seimbang" if a == seimbang else (" ← presisi tinggi" if a == presisi_tinggi else "")
        print(f"{a:>7} |       {v['presisi']:.3f}  {v['recall']:.3f} {v['f1']:.3f}      |"
              f"      {u['presisi']:.3f}  {u['recall']:.3f} {u['f1']:.3f}{tanda}")

    hasil = {"seimbang": seimbang, "presisi_tinggi": presisi_tinggi, "tabel": tabel}
    if simpan:
        os.makedirs(os.path.dirname(AMBANG_FILE), exist_ok=True)
        with open(AMBANG_FILE, "w", encoding="utf-8") as f:
            json.dump({"ambang": seimbang, "pilihan_presisi_tinggi": presisi_tinggi,
                       "dipilih_dari": "split validasi",
                       "hasil_uji_seimbang": tabel[seimbang]["uji"],
                       "hasil_uji_presisi_tinggi":
                           tabel[presisi_tinggi]["uji"] if presisi_tinggi else None},
                      f, indent=2)
        print(f"Ambang {seimbang} disimpan ke {AMBANG_FILE}")
    return hasil


def _data_uji() -> tuple:
    """(teks, label 0/1) split UJI resmi — dari CSV lokal bila ada, jika tidak dari Hub."""
    csv_lokal = os.path.join(PROJECT_DIR, "data", "latih_sarkasme.csv")
    if os.path.isfile(csv_lokal):
        import csv
        teks, y = [], []
        with open(csv_lokal, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    teks.append(row["text"])
                    y.append(1 if row["label"] == "sarkas" else 0)
        return teks, y
    from datasets import load_dataset
    ds = load_dataset("w11wo/twitter_indonesia_sarcastic", split="test")
    return [r["tweet"] for r in ds], [int(r["label"]) for r in ds]


def verifikasi_label(sumber: str = "") -> dict:
    """Uji KEDUA kemungkinan pemetaan LABEL_0/1 pada split uji resmi.

    Pemetaan yang benar akan jauh lebih baik — itu bukti, bukan tebakan.
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score, f1_score

    sumber = sumber or pilih_sumber()
    teks, y = _data_uji()                      # 0 = non-sarcastic, 1 = sarcastic
    pipe = pipeline("text-classification", model=sumber, tokenizer=sumber)
    mentah = [o["label"] for o in pipe(teks, batch_size=32, truncation=True)]

    hasil = {}
    for nama, peta in (("LABEL_1=sarkas", {"LABEL_0": 0, "LABEL_1": 1}),
                       ("LABEL_0=sarkas", {"LABEL_0": 1, "LABEL_1": 0})):
        pred = [peta.get(m, 0) for m in mentah]
        hasil[nama] = {"akurasi": round(accuracy_score(y, pred), 4),
                       "f1_sarkas": round(f1_score(y, pred, pos_label=1), 4)}
    hasil["n_uji"] = len(teks)
    return hasil


def evaluasi_resmi(sumber: str = "") -> dict:
    """Nilai model pada split uji resmi (538 tweet)."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    eng = SarcasmEngine(sumber)
    teks, y01 = _data_uji()
    y = ["sarkas" if v == 1 else "bukan_sarkas" for v in y01]
    pred = [l for l, _ in eng.predict(teks)]
    print(classification_report(y, pred, digits=4, zero_division=0))
    return {"sumber": eng.sumber, "akurasi": round(accuracy_score(y, pred), 4),
            "f1_sarkas": round(f1_score(y, pred, pos_label="sarkas"), 4)}


def analisis_db(db_path: str, limit: int = 1000) -> int:
    from core.storage import Storage
    store = Storage(db_path)
    baris = store.docs_without_field("sarkasme_label", limit)
    if not baris:
        return 0
    eng = SarcasmEngine()
    print(f"[sarkasme] {len(baris)} dokumen — model: {eng.sumber}")
    teks = [f"{r.get('title', '')} {r.get('content', '')}".strip() for r in baris]
    for r, (lab, p) in zip(baris, eng.predict(teks)):
        store.update_fields(r["doc_id"], sarkasme_label=lab, sarkasme_score=p)
    return len(baris)


if __name__ == "__main__":
    import sys
    if "--ambang" in sys.argv:
        pilih_ambang()
    elif "--verifikasi" in sys.argv:
        print(verifikasi_label())
    elif "--evaluasi" in sys.argv:
        print(evaluasi_resmi())
    else:
        eng = SarcasmEngine()
        contoh = ["wah hebat sekali, jalan rusak dibiarkan tiga tahun. prestasi luar biasa",
                  "jalan di depan rumah rusak, mohon segera diperbaiki",
                  "terima kasih pak, harga naik terus bikin hidup makin 'sejahtera'",
                  "harga cabai hari ini turun di pasar induk"]
        for t, (l, p) in zip(contoh, eng.predict(contoh)):
            print(f"  {l:<13} p={p:.3f}  <- {t}")
