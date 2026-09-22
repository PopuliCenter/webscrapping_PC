"""Tandai berita sindikasi — satu rilis yang dimuat ulang banyak media.

Masalahnya nyata untuk media monitoring: satu rilis Antara bisa dimuat 10 media
dengan isi nyaris sama. Tanpa penandaan, volume terlihat 10 kali lipat dan
sentimennya ikut terhitung 10 kali, sehingga satu isu kecil tampak besar.

Yang dilakukan: hitung kemiripan isi antar berita (TF-IDF + kosinus), kelompokkan
yang sangat mirip, lalu tandai SALINAN dengan kolom `duplikat_dari` yang menunjuk
ke berita UTAMA (yang terbit paling awal — sumber aslinya).

Tidak ada yang dihapus. Dashboard tinggal menyembunyikan salinan; kalau perlu,
kamu tetap bisa melihatnya.

Jalankan:
    python tools/dedup_sindikasi.py                 # pratinjau
    python tools/dedup_sindikasi.py --yakin         # tulis ke database
    python tools/dedup_sindikasi.py --ambang 0.8 --yakin
"""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AMBANG = 0.75          # kemiripan kosinus minimum untuk disebut salinan
MAKS_KARAKTER = 2000   # bagian awal berita sudah cukup; ekor sering beda iklan
POTONG = 500           # ukuran blok saat menghitung kemiripan (hemat memori)


def _induk(peta: dict, x: int) -> int:
    """Union-find: cari wakil kelompok."""
    while peta[x] != x:
        peta[x] = peta[peta[x]]
        x = peta[x]
    return x


def _gabung(peta: dict, a: int, b: int):
    ra, rb = _induk(peta, a), _induk(peta, b)
    if ra != rb:
        peta[max(ra, rb)] = min(ra, rb)


def cari_kelompok(teks: list, ambang: float = AMBANG) -> dict:
    """-> {indeks: indeks_wakil} untuk dokumen yang saling mirip."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                          max_features=50000)
    X = vec.fit_transform(t[:MAKS_KARAKTER].lower() for t in teks)
    peta = {i: i for i in range(len(teks))}
    for awal in range(0, X.shape[0], POTONG):
        blok = X[awal:awal + POTONG]
        sim = linear_kernel(blok, X)          # blok x semua, sparse -> padat kecil
        for i in range(sim.shape[0]):
            for j in (sim[i] >= ambang).nonzero()[0]:
                if awal + i != j:
                    _gabung(peta, awal + i, int(j))
    return {i: _induk(peta, i) for i in peta}


def jalankan(db_path: str = "", ambang: float = AMBANG, yakin: bool = False) -> dict:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"),
                              encoding="utf-8")) or {}
    db_path = db_path or os.path.join(PROJECT_DIR, cfg["storage"]["db_path"])
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.row_factory = sqlite3.Row
    baris = conn.execute(
        "SELECT doc_id, title, content, source, published_at, collected_at "
        "FROM documents ORDER BY doc_id").fetchall()
    if not baris:
        print("Database kosong.")
        return {}

    teks = [f"{r['title']} {r['content']}" for r in baris]
    print(f"Dokumen   : {len(baris)}")
    print(f"Ambang    : kemiripan >= {ambang}")
    peta = cari_kelompok(teks, ambang)

    kelompok = {}
    for i, wakil in peta.items():
        kelompok.setdefault(wakil, []).append(i)
    ganda = {w: anggota for w, anggota in kelompok.items() if len(anggota) > 1}

    def kunci_urut(i):
        r = baris[i]
        return (r["published_at"] or r["collected_at"] or "", i)

    tandai, contoh = [], []
    for anggota in ganda.values():
        anggota.sort(key=kunci_urut)          # terbit paling awal = sumber asli
        utama = anggota[0]
        for lain in anggota[1:]:
            tandai.append((baris[utama]["doc_id"], baris[lain]["doc_id"]))
        if len(contoh) < 5:
            contoh.append((utama, anggota[1:]))

    print(f"\nKelompok sindikasi: {len(ganda)}")
    print(f"Berita salinan    : {len(tandai)} (dari {len(baris)} dokumen)")
    for utama, salinan in contoh:
        print(f"\n  UTAMA  [{baris[utama]['source']}] {(baris[utama]['title'] or '')[:62]}")
        for s in salinan[:4]:
            print(f"  salinan[{baris[s]['source']}] {(baris[s]['title'] or '')[:62]}")

    if not yakin:
        print("\nINI PRATINJAU. Tambahkan --yakin untuk menandai di database.")
    else:
        # Hanya menandai, tidak menghapus: salinan tetap bisa ditelusuri.
        conn.executemany(
            "UPDATE documents SET duplikat_dari=? WHERE doc_id=?", tandai)
        conn.execute("UPDATE documents SET duplikat_dari=NULL "
                     "WHERE doc_id NOT IN (SELECT doc_id FROM documents "
                     "WHERE duplikat_dari IS NOT NULL) AND duplikat_dari=''")
        conn.commit()
        sisa = conn.execute("SELECT COUNT(*) FROM documents "
                            "WHERE duplikat_dari IS NULL OR duplikat_dari=''").fetchone()[0]
        print(f"\nSelesai. Ditandai {len(tandai)} salinan; "
              f"{sisa} berita unik tersisa untuk dihitung.")
    conn.close()
    return {"kelompok": len(ganda), "salinan": len(tandai), "total": len(baris)}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = sys.argv[1:]
    amb = float(a[a.index("--ambang") + 1]) if "--ambang" in a else AMBANG
    jalankan(ambang=amb, yakin="--yakin" in a)
