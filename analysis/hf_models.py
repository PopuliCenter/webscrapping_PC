"""Registry model sentimen Indonesia dari HuggingFace + pembanding antar-model.

BAHAYA YANG DIHINDARI: setiap model memakai URUTAN LABEL BERBEDA.
  mdhugol      -> LABEL_0=positive, LABEL_1=neutral, LABEL_2=negative
  w11wo        -> 0=positive, 1=neutral, 2=negative
  ayameRushia  -> 0=Positive, 1=Neutral, 2=Negative
  IndoBERTweet -> 0=Negative, 1=Neutral, 2=Positive   (TERBALIK!)
Salah petakan = hasil sentimen terbalik total. Karena itu modul ini MEMBACA
`id2label` dari config model saat runtime, dan hanya memakai override manual
bila config-nya tidak informatif (LABEL_0/1/2).

Pakai:
    from analysis.hf_models import compare_models, list_model
    hasil = compare_models(teks, label, ["mdhugol", "w11wo", "indobertweet"])
"""
from __future__ import annotations

import os
from typing import Optional

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KANONIK = ("positive", "neutral", "negative")

MODEL_REGISTRY = {
    "mdhugol": {
        "id": "mdhugol/indonesia-bert-sentiment-classification",
        "lokal": os.path.join(_PROJECT, "models", "indobert-sentiment"),
        # config hanya LABEL_0/1/2 -> WAJIB override (sesuai kartu model)
        "label_override": {"LABEL_0": "positive", "LABEL_1": "neutral", "LABEL_2": "negative"},
        "catatan": "Model yang dipakai proyek ini saat ini.",
    },
    "w11wo": {
        "id": "w11wo/indonesian-roberta-base-sentiment-classifier",
        "lokal": os.path.join(_PROJECT, "models", "sentimen-w11wo"),
        "label_override": None,          # config sudah jelas
        "catatan": "Paling populer (±102rb unduhan). RoBERTa, 3 kelas.",
    },
    "ayame": {
        "id": "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
        "label_override": None,
        "catatan": "Dilatih pada dataset SmSA (IndoNLU).",
    },
    "indobertweet": {
        "id": "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis",
        "label_override": None,
        "catatan": "IndoBERTweet — dilatih dari tweet, cocok untuk data X.",
    },
}

_ALIAS = {
    "label_0": None, "label_1": None, "label_2": None,      # tidak informatif
    "positive": "positive", "positif": "positive", "pos": "positive",
    "neutral": "neutral", "netral": "neutral", "net": "neutral",
    "negative": "negative", "negatif": "negative", "neg": "negative",
}


def list_model() -> list:
    return [{"kunci": k, "id": v["id"], "catatan": v["catatan"]}
            for k, v in MODEL_REGISTRY.items()]


def _resolve(raw_label: str, override: Optional[dict]) -> Optional[str]:
    """Ubah label mentah model -> positive/neutral/negative."""
    if override and raw_label in override:
        return override[raw_label]
    return _ALIAS.get(str(raw_label).strip().lower())


def build_pipeline(kunci_atau_id: str, override: Optional[dict] = None):
    """Muat pipeline + petakan labelnya dengan aman.

    -> (pipeline, fungsi_pemeta, id_model)
    Melempar error bila label tidak bisa dipastikan (mencegah hasil terbalik).
    """
    from transformers import pipeline as hf_pipeline, AutoConfig

    cfg = MODEL_REGISTRY.get(kunci_atau_id)
    model_id = cfg["id"] if cfg else kunci_atau_id
    lokal = (cfg or {}).get("lokal")
    if lokal and os.path.isfile(os.path.join(lokal, "config.json")):
        model_id = lokal                     # salinan lokal: jalan tanpa internet
    override = override if override is not None else (cfg or {}).get("label_override")

    conf = AutoConfig.from_pretrained(model_id)
    id2label = getattr(conf, "id2label", {}) or {}

    # pastikan semua label bisa dipetakan
    tak_terpeta = [v for v in id2label.values() if _resolve(v, override) is None]
    if tak_terpeta:
        raise ValueError(
            f"Label model '{model_id}' tidak bisa dipastikan artinya: {tak_terpeta}. "
            f"Sediakan override, mis. "
            f"build_pipeline('{model_id}', override={{'LABEL_0':'positive', ...}}). "
            f"Menebak urutan label berisiko membalik seluruh hasil."
        )

    pipe = hf_pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
    return pipe, (lambda raw: _resolve(raw, override)), model_id


def predict_labels(pipe, pemeta, texts: list, batch_size: int = 16) -> list:
    out = pipe([(t or "")[:512] for t in texts], batch_size=batch_size, truncation=True)
    return [pemeta(o["label"]) or "neutral" for o in out]


def compare_models(texts: list, labels: list, kunci: Optional[list] = None,
                   max_sample: int = 500) -> dict:
    """Bandingkan beberapa model pada data BERLABEL yang sama.

    Hanya kelas yang ada di data yang dinilai. Pakai data berlabel manusia
    (lihat hf_datasets) agar perbandingannya sahih.
    """
    try:
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     confusion_matrix)
    except Exception:
        return {"error": "scikit-learn belum terpasang"}

    kunci = kunci or list(MODEL_REGISTRY)
    texts, labels = list(texts[:max_sample]), list(labels[:max_sample])
    if not texts:
        return {"error": "Tidak ada data uji."}

    kelas = sorted(set(labels))
    hasil = {}
    for k in kunci:
        nama = MODEL_REGISTRY.get(k, {}).get("id", k)
        try:
            pipe, pemeta, model_id = build_pipeline(k)
            pred = predict_labels(pipe, pemeta, texts)
            acc = accuracy_score(labels, pred)
            pr, rc, f1, _ = precision_recall_fscore_support(
                labels, pred, average="macro", zero_division=0, labels=kelas)
            hasil[k] = {
                "model": model_id,
                "accuracy": round(acc, 4),
                "precision": round(pr, 4),
                "recall": round(rc, 4),
                "f1": round(f1, 4),
                "confusion_matrix": confusion_matrix(labels, pred, labels=kelas).tolist(),
                "labels": kelas,
            }
            print(f"  {k:<14} acc={acc:.4f}  f1={f1:.4f}")
        except Exception as e:
            hasil[k] = {"model": nama, "error": str(e)}
            print(f"  {k:<14} GAGAL: {str(e)[:80]}")

    sah = {k: v for k, v in hasil.items() if "f1" in v}
    terbaik = max(sah, key=lambda k: sah[k]["f1"]) if sah else None
    return {"n_uji": len(texts), "kelas": kelas, "hasil": hasil, "terbaik": terbaik}


def tabel_ringkas(res: dict) -> list:
    if "hasil" not in res:
        return []
    baris = [{"Kunci": k, "Model": v.get("model", ""),
              "Accuracy": v.get("accuracy"), "F1": v.get("f1"),
              "Precision": v.get("precision"), "Recall": v.get("recall"),
              "Status": "ok" if "f1" in v else v.get("error", "")[:60]}
             for k, v in res["hasil"].items()]
    baris.sort(key=lambda r: (r["F1"] or 0), reverse=True)
    return baris


def _mcnemar(y, a, b) -> tuple:
    """(kandidat menang, patokan menang, p) — uji McNemar eksak pada data yang sama."""
    from scipy.stats import binomtest
    km = sum(1 for t, x, z in zip(y, a, b) if z == t and x != t)
    pm = sum(1 for t, x, z in zip(y, a, b) if x == t and z != t)
    n = km + pm
    return km, pm, (binomtest(km, n, 0.5).pvalue if n else 1.0)


def _prediksi_biner(pipe, pemeta, teks: list) -> list:
    """Positif/negatif saja: bandingkan P(positive) vs P(negative) (netral diabaikan)."""
    out = pipe([(t or "")[:512] for t in teks], top_k=None, batch_size=32, truncation=True)
    hasil = []
    for per in out:
        sk = {pemeta(o["label"]): o["score"] for o in per}
        hasil.append("positive" if sk.get("positive", 0) >= sk.get("negative", 0) else "negative")
    return hasil


def bandingkan_sentimen(kandidat: str, patokan: str = "w11wo",
                        csv_latih: str = "data/latih_20k.csv", n_prdect: int = 1000) -> dict:
    """Bandingkan model sentimen KANDIDAT vs PATOKAN pada dua data uji.

    (a) data UJI dari CSV latihan — dibangun ulang dengan split yang SAMA seperti saat
        latihan (seed 42), jadi tak pernah dilihat kandidat.
    (b) PRDECT-ID (ulasan produk, domain LAIN) — menguji apakah kandidat memang lebih
        paham sentimen, bukan sekadar hafal gaya teks data latihnya.
    """
    import random
    from sklearn.metrics import accuracy_score, f1_score
    from analysis.finetune_indobert import load_csv, bagi_data

    teks_a, y_a = bagi_data(load_csv(csv_latih))["test"]

    from datasets import load_dataset
    ds = load_dataset("ZakyF/PRDECT-ID", split="train")
    pasangan = [(r["Customer Review"], str(r["Sentiment"]).strip().lower()) for r in ds
                if r.get("Customer Review") and str(r.get("Sentiment")).strip().lower()
                in ("positive", "negative")]
    random.Random(42).shuffle(pasangan)
    teks_b = [t for t, _ in pasangan[:n_prdect]]
    y_b = [l for _, l in pasangan[:n_prdect]]

    pred = {}
    for nama, sumber in (("patokan", patokan), ("kandidat", kandidat)):
        pipe, pemeta, mid = build_pipeline(sumber)
        pred[nama] = {"a": predict_labels(pipe, pemeta, teks_a, batch_size=32),
                      "b": _prediksi_biner(pipe, pemeta, teks_b), "model": mid}

    hasil = {}
    for kode, judul, y in (("a", f"(a) uji data latih — {len(y_a)} teks, 3 kelas", y_a),
                           ("b", f"(b) PRDECT-ID domain lain — {len(y_b)} teks, pos/neg", y_b)):
        print(judul)
        baris = {}
        for nama in ("patokan", "kandidat"):
            pr = pred[nama][kode]
            baris[nama] = {"akurasi": round(accuracy_score(y, pr), 4),
                           "f1": round(f1_score(y, pr, average="macro", zero_division=0), 4)}
            print(f"  {nama:<9} akurasi {baris[nama]['akurasi']:.4f}  F1 makro {baris[nama]['f1']:.4f}")
        km, pm, p = _mcnemar(y, pred["patokan"][kode], pred["kandidat"][kode])
        print(f"  McNemar: kandidat menang {km}, patokan menang {pm}, p = {p:.4g}")
        baris["p"] = p
        baris["kandidat_lebih_baik"] = p < 0.05 and km > pm
        hasil[kode] = baris
    layak = hasil["a"]["kandidat_lebih_baik"] and not (hasil["b"]["p"] < 0.05 and
                                                      hasil["b"]["kandidat"]["akurasi"]
                                                      < hasil["b"]["patokan"]["akurasi"])
    print("KEPUTUSAN:", "kandidat LEBIH BAIK dan tidak memburuk di domain lain — layak dipakai"
          if layak else "belum layak menggantikan patokan")
    hasil["layak"] = layak
    return hasil


if __name__ == "__main__":
    import sys
    if "--bandingkan" in sys.argv:
        bandingkan_sentimen(sys.argv[sys.argv.index("--bandingkan") + 1])
        sys.exit(0)
    from analysis.hf_datasets import load_labeled
    print("Model tersedia:")
    for m in list_model():
        print(f"  {m['kunci']:<14} {m['id']}\n                 {m['catatan']}")
    print("\nMengambil data uji berlabel manusia...")
    teks, label = load_labeled("carant", limit=300, seimbang=True)
    print("\nMembandingkan model:")
    res = compare_models(teks, label)
    print("\nTerbaik:", res.get("terbaik"))
