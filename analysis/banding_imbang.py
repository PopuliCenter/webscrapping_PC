"""Bandingkan cara menangani kelas timpang: SMOTE vs bobot kelas vs koreksi prior.

Pertanyaannya bukan hanya "mana yang F1-nya tertinggi", tapi juga:
**apakah PROPORSI yang dilaporkan masih sahih?** Untuk monitoring gaya Drone
Emprit, "berapa persen negatif hari ini" sama pentingnya dengan "berita mana
yang negatif". Menyeimbangkan data latih membuat model lebih berani memilih
kelas minoritas — bagus untuk MENEMUKAN, tapi bisa merusak HITUNGAN.

Cara uji:
  - Data berlabel manusia (data/latih_20k.csv) dibuat TIMPANG mirip berita
    (netral banyak, negatif sedikit).
  - Data uji memakai distribusi asli yang timpang itu — tidak diseimbangkan,
    tidak di-SMOTE. Kalau data uji ikut diseimbangkan, semua angka jadi palsu.
  - Tiap metode dinilai dengan makro-F1, recall per kelas, dan SELISIH PROPORSI
    (proporsi prediksi - proporsi sebenarnya) sebagai ukuran bias hitungan.

Jalankan:
    python -m analysis.banding_imbang
    python -m analysis.banding_imbang --csv data/latih_20k.csv --n 9000
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Perkiraan komposisi berita: mayoritas netral, negatif paling langka.
PORSI_BERITA = {"neutral": 0.70, "positive": 0.20, "negative": 0.10}


def muat_csv(path: str) -> tuple:
    teks, label = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = (row.get("text") or "").strip()
            l = (row.get("label") or "").strip().lower()
            if t and l:
                teks.append(t)
                label.append(l)
    return teks, label


def buat_timpang(teks: list, label: list, n: int, porsi: dict, seed: int = 42) -> tuple:
    """Ambil sampel sehingga distribusinya mengikuti `porsi`."""
    import random
    rng = random.Random(seed)
    per_kelas = {}
    for t, l in zip(teks, label):
        per_kelas.setdefault(l, []).append(t)
    out_t, out_l = [], []
    for kelas, bagian in porsi.items():
        tersedia = per_kelas.get(kelas, [])
        rng.shuffle(tersedia)
        ambil = min(int(n * bagian), len(tersedia))
        out_t += tersedia[:ambil]
        out_l += [kelas] * ambil
    gabung = list(zip(out_t, out_l))
    rng.shuffle(gabung)
    return [a for a, _ in gabung], [b for _, b in gabung]


def _koreksi_prior(model, X, kelas: list, prior: dict):
    """Logit adjustment: kurangi log-prior agar kelas langka tidak tenggelam.

    Berbeda dari SMOTE, data latih tidak disentuh — hanya keputusan di akhir
    yang digeser, dan pergeserannya eksplisit serta bisa dibatalkan.
    """
    import numpy as np
    logp = model.predict_log_proba(X)
    penyesuai = np.array([np.log(prior[k]) for k in model.classes_])
    return [model.classes_[i] for i in (logp - penyesuai).argmax(axis=1)]


def jalankan(csv_path: str = "", n: int = 9000, seed: int = 42) -> dict:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, recall_score, precision_score

    csv_path = csv_path or os.path.join(PROJECT_DIR, "data", "latih_20k.csv")
    teks, label = muat_csv(csv_path)
    print(f"Sumber       : {os.path.basename(csv_path)} ({len(teks)} baris, "
          f"{dict(Counter(label))})")

    teks, label = buat_timpang(teks, label, n, PORSI_BERITA, seed)
    print(f"Dibuat timpang: {dict(Counter(label))}")

    X_txt, Xu_txt, y, yu = train_test_split(
        teks, label, test_size=0.25, random_state=seed, stratify=label)
    print(f"Latih {len(y)} | Uji {len(yu)} (distribusi uji ASLI: {dict(Counter(yu))})")

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True)
    X, Xu = vec.fit_transform(X_txt), vec.transform(Xu_txt)
    kelas = sorted(set(label))
    prior = {k: Counter(y)[k] / len(y) for k in kelas}
    print(f"Fitur TF-IDF : {X.shape[1]}  |  prior latih: "
          f"{ {k: round(v, 3) for k, v in prior.items()} }\n")

    def nilai(nama: str, pred) -> dict:
        n_uji = len(yu)
        prop_asli = {k: Counter(yu)[k] / n_uji for k in kelas}
        prop_pred = {k: Counter(pred)[k] / n_uji for k in kelas}
        return {
            "metode": nama,
            "f1_makro": round(f1_score(yu, pred, average="macro", zero_division=0), 4),
            "recall": {k: round(v, 3) for k, v in
                       zip(kelas, recall_score(yu, pred, labels=kelas, average=None,
                                               zero_division=0))},
            "presisi": {k: round(v, 3) for k, v in
                        zip(kelas, precision_score(yu, pred, labels=kelas, average=None,
                                                   zero_division=0))},
            # + berarti kelas itu DILAPORKAN LEBIH BANYAK dari kenyataan
            "bias_proporsi": {k: round(prop_pred[k] - prop_asli[k], 3) for k in kelas},
        }

    hasil = []

    m = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    hasil.append(nilai("apa adanya (tanpa penanganan)", m.predict(Xu)))

    try:
        from imblearn.over_sampling import SMOTE
        Xb, yb = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y)
        print(f"SMOTE        : {dict(Counter(y))} -> {dict(Counter(yb))}")
        ms = LogisticRegression(max_iter=1000, random_state=seed).fit(Xb, yb)
        hasil.append(nilai("SMOTE", ms.predict(Xu)))
    except Exception as e:
        print(f"SMOTE dilewati: {e}")

    try:
        from imblearn.over_sampling import RandomOverSampler
        Xr, yr = RandomOverSampler(random_state=seed).fit_resample(X, y)
        mr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xr, yr)
        hasil.append(nilai("oversample acak (salinan)", mr.predict(Xu)))
    except Exception as e:
        print(f"Oversample dilewati: {e}")

    mw = LogisticRegression(max_iter=1000, random_state=seed,
                            class_weight="balanced").fit(X, y)
    hasil.append(nilai("bobot kelas", mw.predict(Xu)))

    hasil.append(nilai("koreksi prior", _koreksi_prior(m, Xu, kelas, prior)))

    lebar = max(len(h["metode"]) for h in hasil) + 1
    print(f"\n{'metode':<{lebar}} {'F1 makro':>9} | recall per kelas "
          f"{'':>14}| bias proporsi (pred - asli)")
    print("-" * (lebar + 78))
    for h in hasil:
        r = " ".join(f"{k[:3]} {h['recall'][k]:.2f}" for k in kelas)
        b = " ".join(f"{k[:3]} {h['bias_proporsi'][k]:+.3f}" for k in kelas)
        print(f"{h['metode']:<{lebar}} {h['f1_makro']:>9.4f} | {r} | {b}")
    print("\nbias proporsi + = kelas itu dilaporkan LEBIH BANYAK dari kenyataan.")
    return {"hasil": hasil, "prior": prior, "n_uji": len(yu)}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    argv = sys.argv[1:]
    csv_path = argv[argv.index("--csv") + 1] if "--csv" in argv else ""
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 9000
    jalankan(csv_path, n)
