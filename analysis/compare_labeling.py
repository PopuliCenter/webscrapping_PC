"""Bandingkan METODE PELABELAN sentimen terhadap label acuan manusia.

Metode yang diuji:
  A. "inset_hitung"  — lexicon InSet dihitung: (jumlah kata positif − negatif).
                        Ini metode yang lazim dipakai di skrip riset Indonesia,
                        DAN mewarisi kelemahannya: stopword removal membuang
                        kata negasi, sehingga "tidak bagus" terbaca "bagus".
  B. "inset_negasi"  — sama, tetapi kata negasi DIPERTAHANKAN dan membalik
                        polaritas kata sesudahnya.
  C. "lexicon_bobot" — LexiconScorer proyek ini (bobot + negasi + penguat).
  D. "indobert"      — model neural.

Tujuan: menunjukkan secara terukur seberapa besar pengaruh penanganan negasi,
dan apakah pelabelan berbasis lexicon layak dipakai sebagai label latih.

Jalankan:
    python -m analysis.compare_labeling
"""
from __future__ import annotations

import os
import re
from typing import Optional

import requests

from .preprocess import Preprocessor, NEGASI_WORDS

INSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "inset")
INSET_URL = ("https://raw.githubusercontent.com/fajri91/InSet/master/{}.tsv")
_TOKEN = re.compile(r"[a-zA-Z']+")


def unduh_inset() -> tuple:
    """Ambil lexicon InSet (fajri91). Disimpan lokal agar sekali unduh."""
    os.makedirs(INSET_DIR, exist_ok=True)
    hasil = {}
    for kutub in ("positive", "negative"):
        path = os.path.join(INSET_DIR, f"{kutub}.tsv")
        if not os.path.isfile(path):
            r = requests.get(INSET_URL.format(kutub), timeout=60)
            r.raise_for_status()
            with open(path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print(f"[inset] diunduh: {kutub}.tsv")
        kata = set()
        with open(path, encoding="utf-8") as f:
            for ln in f:
                bagian = ln.strip().split("\t")
                if bagian and bagian[0]:
                    kata.add(bagian[0].strip().lower())
        hasil[kutub] = kata
    return hasil["positive"], hasil["negative"]


def _label_dari_skor(skor: int) -> str:
    return "positive" if skor > 0 else ("negative" if skor < 0 else "neutral")


def inset_hitung(tokens: list, pos: set, neg: set) -> str:
    """Metode A — hitung kata positif vs negatif, TANPA memahami negasi."""
    skor = sum(1 for t in tokens if t in pos) - sum(1 for t in tokens if t in neg)
    return _label_dari_skor(skor)


def inset_negasi(tokens: list, pos: set, neg: set) -> str:
    """Metode B — sama, tapi negasi membalik polaritas kata sesudahnya."""
    skor = 0
    for i, t in enumerate(tokens):
        nilai = 1 if t in pos else (-1 if t in neg else 0)
        if nilai and i > 0 and tokens[i - 1] in NEGASI_WORDS:
            nilai = -nilai
        skor += nilai
    return _label_dari_skor(skor)


def bandingkan(teks: list, label_acuan: list, pakai_indobert: bool = True,
               maks: int = 1000) -> dict:
    """Uji semua metode terhadap label acuan manusia."""
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    except Exception:
        return {"error": "scikit-learn belum terpasang"}

    teks, label_acuan = list(teks[:maks]), list(label_acuan[:maks])
    pos, neg = unduh_inset()
    kelas = sorted(set(label_acuan))

    # Preprocessing "gaya skrip riset": negasi IKUT terbuang sebagai stopword
    pp_tanpa_negasi = Preprocessor(merge_negation=False,
                                   extra_stopwords=NEGASI_WORDS)
    # Preprocessing proyek ini: negasi dipertahankan
    pp_dgn_negasi = Preprocessor(merge_negation=False)

    hasil = {}

    def nilai(nama, prediksi):
        acc = accuracy_score(label_acuan, prediksi)
        pr, rc, f1, _ = precision_recall_fscore_support(
            label_acuan, prediksi, average="macro", zero_division=0, labels=kelas)
        hasil[nama] = {"accuracy": round(acc, 4), "precision": round(pr, 4),
                       "recall": round(rc, 4), "f1": round(f1, 4)}
        print(f"  {nama:<16} acc={acc:.4f}  f1={f1:.4f}")

    print(f"Menguji {len(teks)} contoh berlabel manusia...")
    tok_a = [pp_tanpa_negasi.tokens(t) for t in teks]
    nilai("inset_hitung", [inset_hitung(tk, pos, neg) for tk in tok_a])

    tok_b = [pp_dgn_negasi.tokens(t) for t in teks]
    nilai("inset_negasi", [inset_negasi(tk, pos, neg) for tk in tok_b])

    from .lexicon_id import LexiconScorer
    ls = LexiconScorer()
    nilai("lexicon_bobot", [ls.score(t)["label"] for t in teks])

    if pakai_indobert:
        try:
            from .sentiment import SentimentEngine
            eng = SentimentEngine({"engine": "indobert"})
            if eng.engine == "indobert":
                pred = [l for l, _ in eng.predict_batch(teks)]
                nilai("indobert", pred)
        except Exception as e:
            print(f"  indobert dilewati: {str(e)[:70]}")

    sah = {k: v for k, v in hasil.items() if "f1" in v}
    return {"n": len(teks), "kelas": kelas, "hasil": hasil,
            "terbaik": max(sah, key=lambda k: sah[k]["f1"]) if sah else None}


if __name__ == "__main__":
    from .hf_datasets import load_labeled
    teks, label = load_labeled("carant", limit=4000, seimbang=True)
    res = bandingkan(teks, label, maks=900)
    print("\nTerbaik:", res.get("terbaik"))
