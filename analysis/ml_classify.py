"""Klasifikasi sentimen: TF-IDF + SMOTE, 6 algoritma, BEFORE vs AFTER.

Metodologi (standar riset sentimen Bahasa Indonesia):
  1. Preprocessing teks (lihat preprocess.py)
  2. Pembobotan TF-IDF
  3. Split data latih/uji (stratified)
  4. Latih 6 classifier pada data ASLI (tidak seimbang)  -> BEFORE
  5. Latih ulang pada data hasil SMOTE (seimbang)        -> AFTER
  6. Bandingkan accuracy / precision / recall / F1 + confusion matrix

Algoritma: Logistic Regression, Decision Tree, Random Forest, SVM,
           K-Nearest Neighbors, Naive Bayes.

Jalankan mandiri:
    python -m analysis.ml_classify
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Optional

from .preprocess import Preprocessor

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, classification_report)
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False


def build_models(random_state: int = 42) -> dict:
    """Enam classifier yang dibandingkan."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree":       DecisionTreeClassifier(random_state=random_state),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM":                 LinearSVC(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         MultinomialNB(),
    }


def load_dataset_from_db(db_path: str, platform: Optional[str] = None,
                         min_per_class: int = 5) -> tuple:
    """Ambil (teks, label) dari kolom sentimen di database."""
    conn = sqlite3.connect(db_path)
    q = ("SELECT title, content, sentiment_label FROM documents "
         "WHERE sentiment_label IS NOT NULL AND sentiment_label != ''")
    params = []
    if platform:
        q += " AND platform=?"
        params.append(platform)
    rows = conn.execute(q, params).fetchall()
    conn.close()

    texts = [f"{(t or '')} {(c or '')}".strip() for t, c, _ in rows]
    labels = [lab for _, _, lab in rows]

    # buang kelas yang terlalu sedikit (tak bisa di-split/SMOTE)
    cnt = Counter(labels)
    keep = {k for k, v in cnt.items() if v >= min_per_class}
    pairs = [(t, l) for t, l in zip(texts, labels) if l in keep]
    if not pairs:
        return [], []
    texts, labels = zip(*pairs)
    return list(texts), list(labels)


def _metrics(y_true, y_pred, labels) -> dict:
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "f1": round(f1, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def run_comparison(texts: list, labels: list, test_size: float = 0.2,
                   random_state: int = 42, max_features: int = 5000,
                   ngram_max: int = 2, p: Optional[Preprocessor] = None) -> dict:
    """Latih & evaluasi 6 model, sebelum dan sesudah SMOTE."""
    if not _HAS_SKLEARN:
        return {"error": "scikit-learn belum terpasang (pip install scikit-learn)"}
    if len(set(labels)) < 2:
        return {"error": "Butuh minimal 2 kelas sentimen untuk klasifikasi."}
    if len(texts) < 20:
        return {"error": f"Data terlalu sedikit ({len(texts)} dokumen). Kumpulkan lebih banyak dulu."}

    pp = p or Preprocessor()
    bersih = [pp.clean(t) for t in texts]

    # buang dokumen yang jadi kosong setelah preprocessing
    pairs = [(t, l) for t, l in zip(bersih, labels) if t.strip()]
    if len(pairs) < 20:
        return {"error": "Terlalu sedikit dokumen tersisa setelah preprocessing."}
    bersih, labels = map(list, zip(*pairs))

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        bersih, labels, test_size=test_size, random_state=random_state, stratify=labels)

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)

    kelas = sorted(set(labels))
    dist_awal = dict(Counter(y_train))

    # ── SMOTE ───────────────────────────────────────────────────
    X_train_bal, y_train_bal, smote_info = X_train, y_train, "tidak dijalankan"
    if _HAS_SMOTE:
        n_min = min(Counter(y_train).values())
        k = min(5, n_min - 1)
        if k >= 1:
            try:
                X_train_bal, y_train_bal = SMOTE(
                    random_state=random_state, k_neighbors=k).fit_resample(X_train, y_train)
                smote_info = f"berhasil (k_neighbors={k})"
            except Exception as e:
                smote_info = f"gagal: {e}"
        else:
            smote_info = f"dilewati (kelas terkecil hanya {n_min} sampel)"
    else:
        smote_info = "imbalanced-learn belum terpasang"

    dist_smote = dict(Counter(y_train_bal))

    hasil = {}
    for nama, _ in build_models(random_state).items():
        baris = {}
        for tahap, Xtr, ytr in (("before", X_train, y_train),
                                ("after", X_train_bal, y_train_bal)):
            model = build_models(random_state)[nama]        # instance baru tiap tahap
            try:
                model.fit(Xtr, ytr)
                baris[tahap] = _metrics(y_test, model.predict(X_test), kelas)
            except Exception as e:
                baris[tahap] = {"error": str(e)}
        # selisih F1 sesudah - sebelum
        try:
            baris["delta_f1"] = round(baris["after"]["f1"] - baris["before"]["f1"], 4)
        except Exception:
            baris["delta_f1"] = None
        hasil[nama] = baris

    return {
        "n_dokumen": len(bersih),
        "n_latih": len(y_train),
        "n_uji": len(y_test),
        "kelas": kelas,
        "distribusi_sebelum_smote": dist_awal,
        "distribusi_sesudah_smote": dist_smote,
        "smote": smote_info,
        "fitur_tfidf": X_train.shape[1],
        "hasil": hasil,
    }


def summary_table(res: dict) -> list:
    """Ubah hasil jadi tabel datar untuk ditampilkan/diekspor."""
    if "hasil" not in res:
        return []
    baris = []
    for nama, d in res["hasil"].items():
        b, a = d.get("before", {}), d.get("after", {})
        baris.append({
            "Model": nama,
            "Acc (before)": b.get("accuracy"), "Acc (after)": a.get("accuracy"),
            "F1 (before)": b.get("f1"),        "F1 (after)": a.get("f1"),
            "Precision (after)": a.get("precision"), "Recall (after)": a.get("recall"),
            "Δ F1": d.get("delta_f1"),
        })
    baris.sort(key=lambda r: (r["F1 (after)"] or 0), reverse=True)
    return baris


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    db = cfg["storage"]["db_path"]
    texts, labels = load_dataset_from_db(db)
    print(f"Dataset: {len(texts)} dokumen, distribusi: {Counter(labels)}")
    res = run_comparison(texts, labels)
    if "error" in res:
        print("Error:", res["error"])
    else:
        print(f"SMOTE: {res['smote']}")
        print(f"Sebelum: {res['distribusi_sebelum_smote']}  ->  Sesudah: {res['distribusi_sesudah_smote']}\n")
        for r in summary_table(res):
            print(f"  {r['Model']:<22} F1 {r['F1 (before)']} -> {r['F1 (after)']}  (Δ {r['Δ F1']})")
