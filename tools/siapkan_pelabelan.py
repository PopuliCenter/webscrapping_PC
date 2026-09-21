"""Siapkan paragraf berita untuk DILABELI MANUSIA -> data/label_berita.csv.

Kenapa per paragraf, bukan per berita: satu berita bisa memuat pernyataan
pemerintah yang netral dan protes warga yang negatif. Melabeli paragraf lebih
mudah disepakati, dan itu pula unit yang dipakai mesin sentimen sekarang.

Dua bagian dengan cara pemilihan BERBEDA — ini penting:

  uji   : sampel ACAK. Dipakai mengukur akurasi di berita sungguhan. Harus acak,
          kalau tidak angkanya tak bisa dipercaya.
  latih : dipilih yang PALING INFORMATIF — dua model berbeda pendapat, atau
          modelnya ragu. Paragraf yang semua model yakin dan sepakat hampir tak
          mengajarkan apa-apa, jadi sayang menghabiskan waktumu di situ.

Usulan label diisikan supaya kamu tinggal membenarkan, bukan mengetik dari nol.
Usulan itu TEBAKAN MESIN — yang menentukan tetap kamu.

Jalankan:
    python tools/siapkan_pelabelan.py
    python tools/siapkan_pelabelan.py --uji 120 --latih 180
"""
from __future__ import annotations

import csv
import os
import random
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Potongan navigasi/iklan yang bukan isi berita.
SAMPAH = re.compile(
    r"baca juga|simak juga|lihat juga|advertisement|scroll to continue|"
    r"saksikan video|copyright|hak cipta|all rights reserved|"
    r"^\s*(sumber|editor|penulis|reporter|foto)\s*[:：]", re.I)
MIN_KATA = 8               # paragraf terlalu pendek: tak cukup konteks untuk dinilai
MAKS_PER_SUMBER = 60       # jangan sampai satu media mendominasi


def _kunci(teks: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", teks.lower())


def kumpulkan_paragraf(db_path: str) -> list:
    """-> [(doc_id, source, url, paragraf)] dari semua berita, sudah disaring."""
    from analysis.sentiment import potong_bagian
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    baris = conn.execute(
        "SELECT doc_id, source, url, title, content FROM documents "
        "WHERE platform='news'").fetchall()
    conn.close()

    keluar, terlihat, per_sumber = [], set(), {}
    for r in baris:
        teks = f"{r['title']}\n\n{r['content']}".strip()
        for par in potong_bagian(teks):
            if len(par.split()) < MIN_KATA or SAMPAH.search(par):
                continue
            k = _kunci(par)
            if not k or k in terlihat:
                continue
            if per_sumber.get(r["source"], 0) >= MAKS_PER_SUMBER:
                continue
            terlihat.add(k)
            per_sumber[r["source"]] = per_sumber.get(r["source"], 0) + 1
            keluar.append((r["doc_id"], r["source"] or "", r["url"] or "", par))
    return keluar


def nilai_dua_model(paragraf: list) -> list:
    """-> [(label_a, skor_a, label_b)] dari model aktif & model pembanding."""
    from analysis.sentiment import SentimentEngine
    aktif = SentimentEngine({"engine": "indobert",
                             "model_dir": "models/indobert-sentiment-finetuned"})
    banding = SentimentEngine({"engine": "indobert",
                               "model_dir": "models/sentimen-w11wo"})
    a = aktif.predict_batch(paragraf)
    b = banding.predict_batch(paragraf)
    return [(la, sa, lb) for (la, sa), (lb, _) in zip(a, b)]


def siapkan(db_path: str = "", n_uji: int = 120, n_latih: int = 180,
            seed: int = 42) -> str:
    db_path = db_path or os.path.join(PROJECT_DIR, "data", "monitoring.db")
    out = os.path.join(PROJECT_DIR, "data", "label_berita.csv")

    data = kumpulkan_paragraf(db_path)
    print(f"Paragraf layak: {len(data)} (dari berita di {os.path.basename(db_path)})")
    if len(data) < n_uji + 20:
        print("PERINGATAN: paragraf terlalu sedikit. Tarik lebih banyak berita dulu "
              "(jalankan run_once.py beberapa kali atau tambah kata kunci).")

    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)

    # 1) UJI: acak murni — inilah yang membuat angka akurasinya sah.
    uji = idx[:min(n_uji, len(idx))]
    sisa = idx[len(uji):]

    # 2) LATIH: dari sisanya, dahulukan yang dua model BERBEDA pendapat atau ragu.
    print("Menilai dengan 2 model (untuk memilih yang paling informatif)...")
    nilai = nilai_dua_model([data[i][3] for i in sisa])
    berbobot = []
    for i, (la, sa, lb) in zip(sisa, nilai):
        beda = 1 if la != lb else 0
        ragu = 1 - abs(sa)                       # |skor| kecil = model ragu
        berbobot.append((beda * 2 + ragu, i, la))
    berbobot.sort(reverse=True)
    latih = [i for _, i, _ in berbobot[:n_latih]]
    usul = {i: la for _, i, la in berbobot}

    nilai_uji = nilai_dua_model([data[i][3] for i in uji])
    for i, (la, _, _) in zip(uji, nilai_uji):
        usul[i] = la

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "bagian", "source", "url", "paragraf", "usulan", "label"])
        for nomor, (i, bagian) in enumerate(
                [(i, "uji") for i in uji] + [(i, "latih") for i in latih], 1):
            doc_id, source, url, par = data[i]
            w.writerow([nomor, bagian, source, url, par, usul.get(i, ""), ""])

    print(f"Tersimpan: {out}")
    print(f"  uji   {len(uji):>4} paragraf (sampel acak — untuk mengukur akurasi)")
    print(f"  latih {len(latih):>4} paragraf (dipilih yang paling informatif)")
    print("\nLangkah berikutnya: buka dashboard -> tab '🏷️ Pelabelan'.")
    return out


def ekspor(label_path: str = "", out: str = "") -> str:
    """Ubah hasil pelabelan -> CSV siap latih (text,label,split).

    'uji' jadi split test (angka akhir dihitung di situ), 'latih' jadi train,
    baris berlabel 'lewati' atau belum dilabeli dibuang.
    """
    label_path = label_path or os.path.join(PROJECT_DIR, "data", "label_berita.csv")
    out = out or os.path.join(PROJECT_DIR, "data", "latih_berita.csv")
    peta = {"uji": "test", "latih": "train"}
    baris = []
    with open(label_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip().lower()
            if lab not in ("positive", "neutral", "negative"):
                continue
            baris.append((r["paragraf"], lab, peta.get(r.get("bagian", ""), "train")))
    if not baris:
        raise SystemExit("Belum ada baris berlabel. Labeli dulu di tab Pelabelan.")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "split"])
        w.writerows(baris)
    from collections import Counter
    for s in ("train", "test"):
        c = Counter(l for _, l, sp in baris if sp == s)
        print(f"  {s:<6} {sum(c.values()):>4}  {dict(c)}")
    print(f"Tersimpan: {out} ({len(baris)} baris)")
    return out


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = sys.argv[1:]
    if "--ekspor" in a:
        ekspor()
        raise SystemExit(0)
    siapkan(n_uji=int(a[a.index("--uji") + 1]) if "--uji" in a else 120,
            n_latih=int(a[a.index("--latih") + 1]) if "--latih" in a else 180)
