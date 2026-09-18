"""Intent & sentimen fine-grained (5 tingkat) TANPA data latih.

Belum ada dataset intent atau sentimen 5-tingkat berbahasa Indonesia yang
relevan untuk topik sosial-politik, jadi dipakai dua pendekatan:

1. ZERO-SHOT (NLI) — model memeriksa apakah teks "mengimplikasikan" sebuah
   hipotesis, mis. "Tujuan teks ini adalah menyampaikan keluhan." Bisa untuk
   label apa pun tanpa melatih, tapi lebih lambat & kurang akurat dari model
   yang dilatih khusus.
     - mdeberta : MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
                  (NLI 26 bahasa termasuk Indonesia; id2label terverifikasi)
     - indonli  : w11wo/indonesian-roberta-base-indonli (NLI asli Indonesia)

2. POLARITAS TERKALIBRASI (khusus fine-grained) — memakai model sentimen
   3-kelas yang sudah bagus, mengambil SELISIH LOGIT positif-negatif, lalu
   memotongnya jadi 5 tingkat. Logit dipakai (bukan probabilitas) karena
   probabilitas model cenderung ~0.99 sehingga hampir semua jatuh ke "sangat".
   Ambang dikalibrasi pada separuh data berlabel, diuji pada separuh lainnya.

Akurasi fine-grained DIUKUR pada PRDECT-ID (rating bintang 1-5), 600 ulasan uji:
  polaritas terkalibrasi : tepat 0.655 | meleset<=1 0.887 | F1 0.437 | Spearman 0.822
  zero-shot mDeBERTa     : tepat 0.172 | meleset<=1 0.938 | F1 0.149 | Spearman 0.822
  -> zero-shot tahu ARAH tapi tak bisa membedakan KEKUATAN; polaritas dipakai.
Akurasi intent BELUM bisa diukur — tidak ada data berlabel yang relevan.

Pakai:
    from analysis.zeroshot import IntentZeroShot, IntensitasPolaritas
    IntentZeroShot().predict(["tolong jelaskan kapan bansos cair?"])
"""
from __future__ import annotations

import json
import os
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMBANG_FILE = os.path.join(PROJECT_DIR, "resources", "ambang_intensitas.json")

ZS_MODELS = {
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "indonli": "w11wo/indonesian-roberta-base-indonli",
}
# Salinan lokal (dibuat dengan tools/unduh_model.py) dipakai lebih dulu.
ZS_LOKAL = {
    "mdeberta": os.path.join(PROJECT_DIR, "models", "zeroshot-mdeberta"),
    "indonli": os.path.join(PROJECT_DIR, "models", "zeroshot-indonli"),
}


def _sumber_zs(model: str) -> str:
    lokal = ZS_LOKAL.get(model)
    if lokal and os.path.isfile(os.path.join(lokal, "config.json")) and any(
            f.endswith(".safetensors") for f in os.listdir(lokal)):
        return lokal
    return ZS_MODELS.get(model, model)


# kunci singkat -> frasa hipotesis (dibaca model sebagai kalimat Indonesia)
INTENT_LABELS = {
    "keluhan":    "menyampaikan keluhan",
    "pertanyaan": "mengajukan pertanyaan",
    "saran":      "memberi saran atau usulan",
    "dukungan":   "memberi dukungan atau pujian",
    "kritik":     "menyampaikan kritik atau penolakan",
    "informasi":  "membagikan informasi atau berita",
    "ajakan":     "mengajak orang lain melakukan sesuatu",
}
TEMPLATE_INTENT = "Tujuan teks ini adalah {}."

TINGKAT = ["sangat negatif", "negatif", "netral", "positif", "sangat positif"]
TEMPLATE_INTENSITAS = "Sentimen teks ini {}."


# ── Zero-shot umum ──────────────────────────────────────────────
class ZeroShot:
    def __init__(self, model: str = "mdeberta"):
        from transformers import pipeline
        self.model_id = _sumber_zs(model)
        self._pipe = pipeline("zero-shot-classification", model=self.model_id,
                              tokenizer=self.model_id)

    def classify(self, texts: list, labels: dict, template: str,
                 batch_size: int = 8) -> list:
        """-> [(kunci_label, skor), ...]. Satu label utama per teks."""
        frasa = list(labels.values())
        balik = {v: k for k, v in labels.items()}
        texts = [(t or "")[:512] for t in texts]
        out = self._pipe(texts, candidate_labels=frasa, hypothesis_template=template,
                         multi_label=False, batch_size=batch_size)
        if isinstance(out, dict):
            out = [out]
        return [(balik[o["labels"][0]], round(float(o["scores"][0]), 4)) for o in out]


class IntentZeroShot(ZeroShot):
    def predict(self, texts: list, batch_size: int = 8) -> list:
        return self.classify(texts, INTENT_LABELS, TEMPLATE_INTENT, batch_size)


class IntensitasZeroShot(ZeroShot):
    def predict(self, texts: list, batch_size: int = 8) -> list:
        return self.classify(texts, {t: t for t in TINGKAT}, TEMPLATE_INTENSITAS, batch_size)


# ── Fine-grained lewat polaritas terkalibrasi ───────────────────
class IntensitasPolaritas:
    """5 tingkat dari selisih logit model sentimen 3-kelas."""

    def __init__(self, model: str = "w11wo", ambang: Optional[list] = None):
        from .hf_models import build_pipeline
        self._pipe, self._pemeta, self.model_id = build_pipeline(model)
        self.ambang = ambang or self._muat_ambang()

    @staticmethod
    def _muat_ambang() -> Optional[list]:
        if os.path.isfile(AMBANG_FILE):
            with open(AMBANG_FILE, encoding="utf-8") as f:
                return json.load(f).get("ambang")
        return None

    def polaritas(self, texts: list, batch_size: int = 16) -> list:
        """Selisih logit (positif - negatif) per teks."""
        texts = [(t or "")[:512] for t in texts]
        out = self._pipe(texts, top_k=None, function_to_apply="none",
                         batch_size=batch_size, truncation=True)
        hasil = []
        for per_teks in out:
            skor = {self._pemeta(o["label"]): float(o["score"]) for o in per_teks}
            hasil.append(skor.get("positive", 0.0) - skor.get("negative", 0.0))
        return hasil

    @staticmethod
    def ke_tingkat(p: float, ambang: list) -> str:
        for batas, nama in zip(ambang, TINGKAT):
            if p <= batas:
                return nama
        return TINGKAT[-1]

    def predict(self, texts: list, batch_size: int = 16) -> list:
        if not self.ambang:
            raise RuntimeError("Ambang belum dikalibrasi. Jalankan: "
                               "python -m analysis.zeroshot --kalibrasi")
        return [(self.ke_tingkat(p, self.ambang), round(p, 4))
                for p in self.polaritas(texts, batch_size)]


def kalibrasi_ambang(polaritas: list, rating: list) -> list:
    """Pilih 4 ambang agar proporsi tiap tingkat menyamai proporsi rating 1-5.

    Dipakai HANYA pada data kalibrasi — bukan data uji.
    """
    import numpy as np
    p = np.sort(np.asarray(polaritas))
    kum = np.cumsum([sum(1 for r in rating if r == k) for k in (1, 2, 3, 4)]) / len(rating)
    return [float(np.quantile(p, q)) for q in kum]


# ── Evaluasi fine-grained pada PRDECT-ID ────────────────────────
def load_prdect(limit: Optional[int] = None) -> tuple:
    """(teks, rating 1-5) dari PRDECT-ID — ulasan produk berlabel bintang."""
    from datasets import load_dataset
    ds = load_dataset("ZakyF/PRDECT-ID", split="train")
    teks, rating = [], []
    for row in ds:
        t = (row.get("Customer Review") or "").strip()
        try:
            r = int(float(row.get("Customer Rating")))
        except (TypeError, ValueError):
            continue
        if t and 1 <= r <= 5:
            teks.append(t)
            rating.append(r)
        if limit and len(teks) >= limit:
            break
    return teks, rating


def evaluasi_intensitas(n: int = 1000, pakai_zeroshot: bool = True,
                        seed: int = 42) -> dict:
    """Bandingkan metode fine-grained terhadap rating bintang (proxy)."""
    import random
    from collections import Counter
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score, f1_score

    teks, rating = load_prdect()
    print(f"PRDECT-ID: {len(teks)} ulasan — rating {dict(sorted(Counter(rating).items()))}")
    idx = list(range(len(teks)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    separuh = len(idx) // 2
    kal, uji = idx[:separuh], idx[separuh:]

    def ukur(nama, pred_tingkat, pred_numerik):
        y = [rating[i] for i in uji]
        yp = [TINGKAT.index(t) + 1 for t in pred_tingkat]
        dekat = sum(abs(a - b) <= 1 for a, b in zip(y, yp)) / len(y)
        rho = spearmanr(y, pred_numerik).statistic
        hasil = {"akurasi_tepat": round(accuracy_score(y, yp), 4),
                 "f1_makro": round(f1_score(y, yp, average="macro", zero_division=0), 4),
                 "akurasi_selisih_1": round(dekat, 4),
                 "spearman": round(float(rho), 4)}
        print(f"  {nama:<22} tepat={hasil['akurasi_tepat']:.3f}  ±1={dekat:.3f}  "
              f"F1={hasil['f1_makro']:.3f}  spearman={rho:.3f}")
        return hasil

    res = {"n_kalibrasi": len(kal), "n_uji": len(uji)}

    print("\n[1] Polaritas terkalibrasi (w11wo)")
    ip = IntensitasPolaritas("w11wo", ambang=[0, 0, 0, 0])
    pol_kal = ip.polaritas([teks[i] for i in kal])
    ambang = kalibrasi_ambang(pol_kal, [rating[i] for i in kal])
    print(f"  ambang hasil kalibrasi: {[round(a, 3) for a in ambang]}")
    pol_uji = ip.polaritas([teks[i] for i in uji])
    res["polaritas"] = ukur("polaritas_kalibrasi",
                            [IntensitasPolaritas.ke_tingkat(p, ambang) for p in pol_uji],
                            pol_uji)
    res["ambang"] = ambang

    if pakai_zeroshot:
        print("\n[2] Zero-shot NLI (mDeBERTa)")
        zs = IntensitasZeroShot("mdeberta")
        pred = zs.predict([teks[i] for i in uji])
        tingkat = [t for t, _ in pred]
        res["zeroshot"] = ukur("zeroshot_mdeberta", tingkat,
                               [TINGKAT.index(t) for t in tingkat])

    kandidat = {k: v for k, v in res.items() if isinstance(v, dict)}
    res["terbaik"] = max(kandidat, key=lambda k: kandidat[k]["spearman"]) if kandidat else None
    return res


def simpan_ambang(ambang: list, info: dict):
    os.makedirs(os.path.dirname(AMBANG_FILE), exist_ok=True)
    with open(AMBANG_FILE, "w", encoding="utf-8") as f:
        json.dump({"ambang": ambang, **info}, f, indent=2, ensure_ascii=False)
    print(f"Ambang disimpan ke {AMBANG_FILE}")


# ── Isi database ────────────────────────────────────────────────
def analisis_db(db_path: str, tugas: tuple = ("intent", "intensitas"),
                limit: int = 500) -> dict:
    from core.storage import Storage
    store = Storage(db_path)
    hasil = {}

    if "intent" in tugas:
        baris = store.docs_without_field("intent_label", limit)
        if baris:
            print(f"[intent] {len(baris)} dokumen (zero-shot, agak lambat di CPU)...")
            zs = IntentZeroShot()
            teks = [f"{r.get('title', '')} {r.get('content', '')}".strip() for r in baris]
            for r, (lab, skor) in zip(baris, zs.predict(teks)):
                store.update_fields(r["doc_id"], intent_label=lab, intent_score=skor)
        hasil["intent"] = len(baris)

    if "intensitas" in tugas:
        baris = store.docs_without_field("intensitas_label", limit)
        if baris:
            ip = IntensitasPolaritas()
            if not ip.ambang:
                print("[intensitas] ambang belum dikalibrasi — "
                      "jalankan: python -m analysis.zeroshot --kalibrasi")
                hasil["intensitas"] = 0
                return hasil
            print(f"[intensitas] {len(baris)} dokumen...")
            teks = [f"{r.get('title', '')} {r.get('content', '')}".strip() for r in baris]
            for r, (lab, p) in zip(baris, ip.predict(teks)):
                store.update_fields(r["doc_id"], intensitas_label=lab, intensitas_score=p)
        hasil["intensitas"] = len(baris)
    return hasil


if __name__ == "__main__":
    import sys
    if "--kalibrasi" in sys.argv:
        r = evaluasi_intensitas(n=1000, pakai_zeroshot="--tanpa-zeroshot" not in sys.argv)
        simpan_ambang(r["ambang"], {"sumber": "PRDECT-ID", "model": "w11wo",
                                    "evaluasi": r.get("polaritas")})
        print("\nTerbaik:", r["terbaik"])
    else:
        contoh = ["tolong jelaskan kapan bansos cair?",
                  "jalan di depan rumah rusak parah sudah 3 bulan tidak diperbaiki",
                  "mari kita dukung program ini bersama-sama!",
                  "pemerintah resmi menaikkan harga BBM mulai besok pukul 14.00",
                  "sebaiknya anggaran dialihkan ke pendidikan saja"]
        for t, (lab, s) in zip(contoh, IntentZeroShot().predict(contoh)):
            print(f"  {lab:<11} {s:.3f}  <- {t}")
