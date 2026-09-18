"""Penilaian sentimen PER-KATA berbobot untuk Bahasa Indonesia.

Berbeda dari IndoBERT (akurat tapi "kotak hitam"), modul ini menghasilkan
rincian kontribusi tiap kata sehingga hasil bisa DIAUDIT dan dijelaskan —
penting untuk laporan riset.

Menangani:
  - bobot kata (lexicon InSet: -5..+5, atau bawaan)
  - negasi     ("tidak bagus"  -> membalik polaritas)
  - penguat    ("sangat bagus" -> memperkuat bobot)
  - peredam    ("agak bagus"   -> memperlemah bobot)

Pakai:
    from analysis.lexicon_id import LexiconScorer
    s = LexiconScorer()
    s.score("pelayanan tidak bagus sama sekali")
    # -> {'label': 'negative', 'score': -0.6, 'details': [...]}
"""
from __future__ import annotations

import os
import re
from typing import Optional

# ── Lexicon berbobot bawaan (dipakai bila InSet tidak tersedia) ──
# Skala -5 (sangat negatif) .. +5 (sangat positif)
LEXICON_BOBOT = {
    # positif
    "bagus": 3, "baik": 3, "hebat": 4, "mantap": 4, "luar biasa": 5, "puas": 3,
    "senang": 3, "suka": 3, "sukses": 4, "berhasil": 4, "untung": 3, "maju": 3,
    "dukung": 3, "apresiasi": 4, "bangga": 4, "optimis": 3, "aman": 3, "damai": 4,
    "adil": 4, "jujur": 4, "bersih": 3, "cepat": 2, "mudah": 2, "murah": 2,
    "nyaman": 3, "indah": 3, "cerdas": 3, "tepat": 3, "unggul": 4, "juara": 4,
    "tumbuh": 2, "naik": 2, "meningkat": 2, "positif": 3, "setuju": 2, "terima": 1,
    "harap": 1, "semangat": 3, "peduli": 3, "ramah": 3, "hormat": 3,
    # negatif
    "buruk": -3, "jelek": -3, "gagal": -4, "rugi": -3, "marah": -4, "kecewa": -4,
    "tolak": -3, "protes": -3, "korupsi": -5, "curang": -5, "bohong": -4,
    "tipu": -4, "turun": -2, "anjlok": -4, "krisis": -4, "masalah": -2,
    "negatif": -3, "kritik": -2, "bahaya": -4, "konflik": -3, "mahal": -2,
    "lambat": -2, "skandal": -5, "hancur": -4, "rusak": -3, "benci": -5,
    "bodoh": -4, "malas": -3, "sulit": -2, "susah": -2, "parah": -3,
    "kacau": -3, "ricuh": -4, "tegang": -2, "khawatir": -2, "takut": -3,
    "sedih": -3, "kesal": -3, "muak": -4, "jengkel": -3, "zalim": -5,
    "sengsara": -4, "miskin": -3, "menderita": -4, "ancam": -4, "serang": -3,
}

NEGASI  = {"tidak", "bukan", "tak", "tanpa", "kurang", "belum", "jangan", "gak", "nggak"}
PENGUAT = {"sangat": 1.6, "sekali": 1.5, "banget": 1.6, "amat": 1.5, "paling": 1.7,
           "sungguh": 1.5, "benar": 1.3, "terlalu": 1.4, "sangat sangat": 2.0,
           "luar": 1.5, "super": 1.6, "makin": 1.3, "semakin": 1.3}
PEREDAM = {"agak": 0.6, "sedikit": 0.5, "lumayan": 0.7, "cukup": 0.7,
           "hampir": 0.6, "kadang": 0.6}

_TOKEN = re.compile(r"[a-zA-Zà-ÿ']+")


def _load_inset(inset_dir: str) -> dict:
    """Muat lexicon InSet (positive.tsv / negative.tsv, format: kata<TAB>bobot)."""
    lex = {}
    if not inset_dir or not os.path.isdir(inset_dir):
        return lex
    for fname in ("positive.tsv", "negative.tsv"):
        path = os.path.join(inset_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    parts = ln.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            lex[parts[0].strip().lower()] = float(parts[1])
                        except ValueError:
                            continue
        except Exception:
            pass
    return lex


class LexiconScorer:
    def __init__(self, inset_dir: str = "", extra: Optional[dict] = None):
        self.lex = dict(LEXICON_BOBOT)
        self.lex.update(_load_inset(inset_dir))
        if extra:
            self.lex.update(extra)
        # skala maksimum untuk normalisasi
        self._max_abs = max((abs(v) for v in self.lex.values()), default=5) or 5

    def score(self, text: str) -> dict:
        """Skor sentimen + rincian kontribusi tiap kata."""
        toks = [t.lower() for t in _TOKEN.findall(text or "")]
        details, total = [], 0.0

        for i, tok in enumerate(toks):
            base = self.lex.get(tok)
            if base is None:
                continue

            faktor, alasan = 1.0, []
            # lihat 2 kata sebelumnya untuk negasi / penguat / peredam
            for j in (i - 1, i - 2):
                if j < 0:
                    continue
                prev = toks[j]
                if prev in NEGASI:
                    faktor *= -1.0
                    alasan.append(f"negasi '{prev}'")
                elif prev in PENGUAT:
                    faktor *= PENGUAT[prev]
                    alasan.append(f"penguat '{prev}'")
                elif prev in PEREDAM:
                    faktor *= PEREDAM[prev]
                    alasan.append(f"peredam '{prev}'")

            nilai = base * faktor
            total += nilai
            details.append({
                "kata": tok,
                "bobot_dasar": base,
                "faktor": round(faktor, 2),
                "kontribusi": round(nilai, 2),
                "alasan": ", ".join(alasan) or "-",
            })

        # normalisasi ke rentang -1..1 (dibagi akar jumlah token agar teks
        # panjang tidak otomatis ekstrem)
        n = max(len(toks), 1)
        norm = total / (self._max_abs * (n ** 0.5)) if details else 0.0
        norm = max(-1.0, min(1.0, norm))
        label = "positive" if norm > 0.05 else ("negative" if norm < -0.05 else "neutral")

        return {
            "label": label,
            "score": round(norm, 4),
            "total_mentah": round(total, 2),
            "jumlah_kata_bersentimen": len(details),
            "details": details,
        }
