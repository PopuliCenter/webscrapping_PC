"""Statistik teks: frekuensi kata, n-gram, dan term khas per kelompok.

Dipakai untuk word cloud, grafik top terms, dan membandingkan kosakata
antar sentimen/topik.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

from .preprocess import Preprocessor


def word_freq(texts: Iterable[str], p: Optional[Preprocessor] = None,
              top_n: int = 50, per_dokumen: bool = False) -> list:
    """Frekuensi kata setelah preprocessing. -> [(kata, jumlah), ...]

    `per_dokumen=True` menghitung tiap kata SEKALI per dokumen (document
    frequency). Ini penting untuk media monitoring: satu artikel panjang yang
    mengulang sebuah kata puluhan kali bisa mendominasi word cloud dan membuat
    isu kecil terlihat besar. Contoh nyata: kata "kpk" muncul 92 kali padahal
    hanya berasal dari SATU berita daftar OTT.
    """
    pp = p or Preprocessor()
    c = Counter()
    for t in texts:
        tok = pp.tokens(t)
        c.update(set(tok) if per_dokumen else tok)
    return c.most_common(top_n)


def ngrams(texts: Iterable[str], n: int = 2, p: Optional[Preprocessor] = None,
           top_n: int = 30) -> list:
    """Frekuensi n-gram (bigram/trigram) -> [("kata1 kata2", jumlah), ...]"""
    pp = p or Preprocessor()
    c = Counter()
    for t in texts:
        toks = pp.tokens(t)
        if len(toks) < n:
            continue
        c.update(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return c.most_common(top_n)


def freq_dict(texts: Iterable[str], p: Optional[Preprocessor] = None,
              per_dokumen: bool = False) -> dict:
    """Kamus {kata: jumlah} — format yang dibutuhkan WordCloud.

    `per_dokumen=True`: satu kata dihitung sekali per dokumen (lihat word_freq).
    """
    pp = p or Preprocessor()
    c = Counter()
    for t in texts:
        tok = pp.tokens(t)
        c.update(set(tok) if per_dokumen else tok)
    return dict(c)


def distinctive_terms(group_texts: Iterable[str], other_texts: Iterable[str],
                      p: Optional[Preprocessor] = None, top_n: int = 20) -> list:
    """Kata yang KHAS pada satu kelompok dibanding kelompok lain.

    Memakai rasio frekuensi relatif (dengan smoothing) — berguna untuk melihat
    kata yang membedakan sentimen negatif vs positif.
    -> [(kata, skor_kekhasan, jumlah_di_kelompok), ...]
    """
    pp = p or Preprocessor()
    a, b = Counter(), Counter()
    for t in group_texts:
        a.update(pp.tokens(t))
    for t in other_texts:
        b.update(pp.tokens(t))

    total_a = sum(a.values()) or 1
    total_b = sum(b.values()) or 1
    out = []
    for w, cnt in a.items():
        if cnt < 2:                       # abaikan kata sangat jarang
            continue
        ra = cnt / total_a
        rb = (b.get(w, 0) + 1) / (total_b + 1)   # smoothing
        out.append((w, round(ra / rb, 3), cnt))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]
