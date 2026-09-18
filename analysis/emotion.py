"""Analisis EMOSI Bahasa Indonesia (bukan sekadar positif/negatif).

Model: StevenLimcorn/indonesian-roberta-base-emotion-classifier
Label (terverifikasi dari config): sadness, anger, love, fear, happy
  -> ditampilkan sebagai: sedih, marah, cinta, takut, senang

Kenapa berguna: dua berita sama-sama "negatif" bisa sangat berbeda —
yang satu memicu KEMARAHAN (berpotensi viral/ricuh), yang lain KESEDIHAN.
Drone Emprit membedakan keduanya.

Pakai:
    from analysis.emotion import EmotionEngine
    e = EmotionEngine()
    e.predict("saya sangat marah dengan kebijakan ini")   # -> ('marah', 0.97)
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

HUB_MODEL = "StevenLimcorn/indonesian-roberta-base-emotion-classifier"
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "indoroberta-emotion")

# Label model (Inggris) -> istilah Indonesia yang dipakai di dashboard
EMOSI_ID = {
    "sadness": "sedih",
    "anger": "marah",
    "love": "cinta",
    "fear": "takut",
    "happy": "senang",
    "joy": "senang",
    "surprise": "terkejut",
    "disgust": "jijik",
}

SEMUA_EMOSI = ["marah", "takut", "sedih", "senang", "cinta"]


def _is_local_model(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(path, f))
               for f in ("model.safetensors", "pytorch_model.bin"))


class EmotionEngine:
    """Klasifikasi emosi. Memakai model lokal bila ada, jika tidak unduh dari Hub."""

    def __init__(self, model_dir: str = "", model_id: str = ""):
        self._pipe = None
        self.tersedia = False
        self.model_source = ""

        model_dir = model_dir or DEFAULT_MODEL_DIR
        sumber = model_dir if _is_local_model(model_dir) else (model_id or HUB_MODEL)
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=sumber, tokenizer=sumber)
            self.model_source = sumber
            self.tersedia = True
        except Exception as e:
            print(f"[emosi] model tidak dapat dimuat ({str(e)[:90]}). "
                  f"Siapkan dengan: python tools/setup_local_models.py --emotion")

    @staticmethod
    def _ke_indonesia(label: str) -> str:
        return EMOSI_ID.get(str(label).strip().lower(), str(label).lower())

    def predict(self, text: str) -> Tuple[str, Optional[float]]:
        if not self.tersedia:
            return "", None
        try:
            out = self._pipe((text or "")[:512], truncation=True)[0]
            return self._ke_indonesia(out["label"]), round(float(out["score"]), 4)
        except Exception:
            return "", None

    def predict_batch(self, texts: list, batch_size: int = 16) -> list:
        """-> [(emosi, skor), ...] sejajar dengan urutan masukan."""
        if not self.tersedia or not texts:
            return [("", None)] * len(texts)
        try:
            outs = self._pipe([(t or "")[:512] for t in texts],
                              batch_size=batch_size, truncation=True)
            return [(self._ke_indonesia(o["label"]), round(float(o["score"]), 4))
                    for o in outs]
        except Exception:
            return [self.predict(t) for t in texts]


def analisis_db(db_path: str, limit: int = 1000, model_dir: str = "") -> int:
    """Isi kolom emosi untuk dokumen yang belum dianalisis. -> jumlah diproses."""
    from core.storage import Storage
    store = Storage(db_path)
    baris = store.docs_without_emotion(limit=limit)
    if not baris:
        print("[emosi] tidak ada dokumen baru untuk dianalisis.")
        return 0

    eng = EmotionEngine(model_dir=model_dir)
    if not eng.tersedia:
        return 0

    print(f"[emosi] menganalisis {len(baris)} dokumen...")
    teks = [f"{r.get('title', '')} {r.get('content', '')}".strip() for r in baris]
    for r, (emo, skor) in zip(baris, eng.predict_batch(teks)):
        if emo:
            store.update_emotion(r["doc_id"], emo, skor)
    return len(baris)


if __name__ == "__main__":
    e = EmotionEngine()
    if e.tersedia:
        contoh = ["saya sangat marah dengan kebijakan yang tidak adil ini",
                  "turut berduka, sedih sekali mendengar kabar ini",
                  "alhamdulillah senang sekali programnya berhasil",
                  "khawatir dan takut harga terus naik"]
        for t in contoh:
            print(f"  {e.predict(t)}  <- {t[:50]}")
