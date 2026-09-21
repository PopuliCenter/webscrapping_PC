"""Analisis sentimen Bahasa Indonesia.

Dua mesin:
  - "lexicon"  : cepat, ringan, tanpa download. Pakai InSet bila tersedia,
                 atau kamus bawaan kecil sebagai fallback.
  - "indobert" : pakai model transformer (akurat, perlu transformers+torch).

API: SentimentEngine(cfg).predict(text) -> (label, score)
     label in {positive, neutral, negative}; score in [-1, 1].
"""
from __future__ import annotations

import os
import re
from typing import Tuple

_TOKEN = re.compile(r"[a-zA-Zà-ÿ']+")

# ── Lokasi model IndoBERT ───────────────────────────────────────
HUB_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "indobert-sentiment")


# Batas aman satu potong teks. IndoBERT memproses maksimal 512 token; 512
# KARAKTER selalu di bawah batas itu, jadi tidak ada teks yang diam-diam terpotong.
MAKS_KARAKTER = 512
# Ambang label gabungan: |skor| di bawah ini dianggap netral (sama dengan lexicon).
AMBANG_GABUNG = 0.05
# Potongan lebih pendek dari ini ditempel ke tetangganya (bukan paragraf utuh).
MIN_GABUNG = 80

_AKHIR_KALIMAT = re.compile(r"(?<=[.!?])\s+")


def potong_bagian(text: str, maks: int = MAKS_KARAKTER,
                  min_gabung: int = MIN_GABUNG) -> list:
    """Pecah teks jadi bagian <= `maks` karakter, satu paragraf satu bagian.

    Paragraf utuh sengaja TIDAK digabung: kalau digabung, paragraf protes bisa
    menempel pada paragraf netral dan nadanya saling menghapus. Hanya potongan
    sangat pendek (< `min_gabung` karakter — baris tanggal, nama kota, sisa
    kutipan) yang ditempel ke tetangganya. Paragraf kelewat panjang dipecah per
    kalimat, dan kalimat raksasa (teks tanpa titik) dipotong keras per `maks`.
    """
    teks = (text or "").strip()
    if not teks:
        return []
    kasar = []
    for par in re.split(r"\n\s*\n|\n", teks):
        par = par.strip()
        if not par:
            continue
        if len(par) <= maks:
            kasar.append(par)
            continue
        buf = ""
        for kal in _AKHIR_KALIMAT.split(par):
            while len(kal) > maks:                 # kalimat tanpa titik
                kasar.append(kal[:maks])
                kal = kal[maks:]
            if len(buf) + len(kal) + 1 <= maks:
                buf = f"{buf} {kal}".strip()
            else:
                if buf:
                    kasar.append(buf)
                buf = kal
        if buf:
            kasar.append(buf)

    bagian = []                                    # tempel potongan sangat pendek
    for p in kasar:
        if (bagian and min(len(bagian[-1]), len(p)) < min_gabung
                and len(bagian[-1]) + len(p) + 1 <= maks):
            bagian[-1] = f"{bagian[-1]} {p}"
        else:
            bagian.append(p)
    return bagian


_LABEL_ALIAS = {
    "label_0": "positive", "label_1": "neutral", "label_2": "negative",
    "positive": "positive", "neutral": "neutral", "negative": "negative",
    "positif": "positive", "netral": "neutral", "negatif": "negative",
    "pos": "positive", "net": "neutral", "neg": "negative",
}


def _map_label(raw: str) -> str:
    """Terima label model bawaan (LABEL_0/1/2) maupun hasil fine-tuning sendiri."""
    return _LABEL_ALIAS.get(str(raw).strip().lower(), "neutral")


def _perangkat() -> int:
    """0 = GPU pertama, -1 = CPU (dipakai transformers.pipeline)."""
    try:
        import torch  # type: ignore
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _is_local_model(path: str) -> bool:
    """Folder model valid bila berisi config.json + bobot model."""
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(path, f))
               for f in ("model.safetensors", "pytorch_model.bin"))

# Kamus bawaan minimal (fallback bila InSet tidak ada). Perluas sesuai kebutuhan.
_POS = {
    "baik", "bagus", "hebat", "mantap", "sukses", "untung", "senang", "puas",
    "dukung", "maju", "berhasil", "naik", "tumbuh", "apresiasi", "positif",
    "setuju", "bangga", "optimis", "aman", "damai", "adil",
}
_NEG = {
    "buruk", "jelek", "gagal", "rugi", "marah", "kecewa", "tolak", "protes",
    "korupsi", "turun", "anjlok", "krisis", "masalah", "bohong", "negatif",
    "kritik", "bahaya", "konflik", "mahal", "lambat", "curang", "skandal",
}
_NEGATORS = {"tidak", "bukan", "tak", "tanpa", "kurang", "belum"}


class SentimentEngine:
    def __init__(self, cfg: dict):
        self.engine = (cfg or {}).get("engine", "lexicon").lower()
        self._pos, self._neg = set(_POS), set(_NEG)
        self._pipe = None

        if self.engine == "lexicon":
            self._load_inset((cfg or {}).get("inset_dir", ""))
        elif self.engine == "indobert":
            self._init_indobert((cfg or {}).get("model_dir", ""))
        else:
            print(f"[sentiment] engine '{self.engine}' tak dikenal -> lexicon")
            self.engine = "lexicon"

    # ── Lexicon ──────────────────────────────────────────────
    def _load_inset(self, inset_dir: str):
        if not inset_dir or not os.path.isdir(inset_dir):
            return
        for fname, target in (("positive.tsv", self._pos), ("negative.tsv", self._neg)):
            path = os.path.join(inset_dir, fname)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            w = line.split("\t")[0].strip().lower()
                            if w:
                                target.add(w)
                except Exception as e:
                    print(f"[sentiment] gagal baca {path}: {e}")

    def _lexicon_predict(self, text: str) -> Tuple[str, float]:
        toks = [t.lower() for t in _TOKEN.findall(text or "")]
        if not toks:
            return "neutral", 0.0
        score = 0
        for i, t in enumerate(toks):
            val = 1 if t in self._pos else (-1 if t in self._neg else 0)
            if val and i > 0 and toks[i - 1] in _NEGATORS:
                val = -val          # negasi membalik polaritas
            score += val
        norm = max(-1.0, min(1.0, score / (len(toks) ** 0.5)))
        label = "positive" if norm > 0.05 else ("negative" if norm < -0.05 else "neutral")
        return label, round(norm, 4)

    # ── IndoBERT ─────────────────────────────────────────────
    def _init_indobert(self, model_dir: str = ""):
        """Muat IndoBERT. Prioritas: folder LOKAL -> baru unduh dari HuggingFace.

        Model lokal (models/indobert-sentiment) membuat program jalan tanpa
        internet dan bisa diganti hasil fine-tuning sendiri.
        """
        model_dir = model_dir or DEFAULT_MODEL_DIR
        if not os.path.isabs(model_dir):
            # relatif terhadap folder PROYEK, bukan folder tempat program dijalankan —
            # kalau tidak, dari folder lain diam-diam kembali ke model Hub
            model_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_dir))
        sumber = model_dir if _is_local_model(model_dir) else HUB_MODEL
        if sumber == HUB_MODEL:
            print(f"[sentiment] model lokal tak ditemukan di '{model_dir}' "
                  f"-> memakai HuggingFace. Jalankan: python tools/setup_local_models.py")
        try:
            from transformers import pipeline  # type: ignore
            # Penilaian per paragraf = jauh lebih banyak panggilan model, jadi pakai
            # GPU bila ada (pipeline bawaannya CPU). Tanpa GPU tetap jalan normal.
            self._pipe = pipeline("sentiment-analysis", model=sumber, tokenizer=sumber,
                                  device=_perangkat())
            self.model_source = sumber
        except Exception as e:
            print(f"[sentiment] IndoBERT gagal dimuat ({e}) -> fallback lexicon")
            self.engine = "lexicon"

    def _indobert_predict(self, text: str) -> Tuple[str, float]:
        try:
            out = self._pipe((text or "")[:512])[0]
            label = _map_label(out["label"])
            conf = float(out["score"])
            signed = conf if label == "positive" else (-conf if label == "negative" else 0.0)
            return label, round(signed, 4)
        except Exception:
            return self._lexicon_predict(text)

    def _indobert_predict_batch(self, texts: list) -> list:
        clipped = [(t or "")[:512] for t in texts]
        try:
            outs = self._pipe(clipped, batch_size=16, truncation=True)
        except Exception:
            return [self._lexicon_predict(t) for t in texts]
        res = []
        for out in outs:
            label = _map_label(out["label"])
            conf = float(out["score"])
            signed = conf if label == "positive" else (-conf if label == "negative" else 0.0)
            res.append((label, round(signed, 4)))
        return res

    # ── Teks panjang: per bagian ─────────────────────────────
    def predict_panjang(self, text: str, maks: int = MAKS_KARAKTER) -> dict:
        """Nilai teks panjang PER BAGIAN, lalu gabungkan.

        `predict()` hanya melihat ~512 karakter pertama (batas IndoBERT), jadi
        berita panjang praktis dinilai dari paragraf pembuka saja. Di sini teks
        dipotong per paragraf/kalimat, tiap bagian dinilai sendiri, dan hasilnya
        digabung dengan bobot panjang bagian (paragraf panjang lebih berpengaruh).

        -> {label, score, bagian_total, bagian_negatif, bagian_positif,
            kutipan_negatif, bagian: [(teks, label, score)]}
        """
        bagian = potong_bagian(text, maks)
        return self._gabung(bagian, self.predict_batch(bagian) if bagian else [])

    def predict_panjang_batch(self, texts: list, maks: int = MAKS_KARAKTER) -> list:
        """predict_panjang untuk banyak teks — semua bagian dinilai sekali jalan."""
        potongan = [potong_bagian(t, maks) for t in texts]
        rata = [b for p in potongan for b in p]
        nilai = iter(self.predict_batch(rata)) if rata else iter(())
        hasil = []
        for p in potongan:
            sub = [next(nilai) for _ in p]
            hasil.append(self._gabung(p, sub))
        return hasil

    @staticmethod
    def _gabung(bagian: list, nilai: list) -> dict:
        if not bagian:
            return {"label": "neutral", "score": 0.0, "bagian_total": 0,
                    "bagian_negatif": 0, "bagian_positif": 0,
                    "kutipan_negatif": "", "bagian": []}
        bobot = [len(b) for b in bagian]
        skor = sum(s * w for (_, s), w in zip(nilai, bobot)) / (sum(bobot) or 1)
        neg = [(s, b) for b, (l, s) in zip(bagian, nilai) if l == "negative"]
        return {"label": ("positive" if skor > AMBANG_GABUNG else
                          "negative" if skor < -AMBANG_GABUNG else "neutral"),
                "score": round(skor, 4), "bagian_total": len(bagian),
                "bagian_negatif": len(neg),
                "bagian_positif": sum(1 for l, _ in nilai if l == "positive"),
                "kutipan_negatif": min(neg)[1][:300] if neg else "",
                "bagian": [(b, l, s) for b, (l, s) in zip(bagian, nilai)]}

    # ── Publik ───────────────────────────────────────────────
    def predict(self, text: str) -> Tuple[str, float]:
        if self.engine == "indobert" and self._pipe is not None:
            return self._indobert_predict(text)
        return self._lexicon_predict(text)

    def predict_batch(self, texts: list) -> list:
        """Prediksi banyak teks sekaligus (jauh lebih cepat untuk IndoBERT)."""
        if not texts:
            return []
        if self.engine == "indobert" and self._pipe is not None:
            return self._indobert_predict_batch(texts)
        return [self._lexicon_predict(t) for t in texts]
