"""Rapikan database: label ulang kata kunci & hapus dokumen topik lama.

Database menumpuk semua penarikan sebelumnya. Setelah berganti topik, dokumen
lama membuat word cloud, heatmap, dan distribusi sentimen jadi menyesatkan.
Alat ini membersihkannya — dengan dua pengaman:

  1. CADANGAN otomatis dibuat sebelum apa pun dihapus.
  2. PRATINJAU dulu (bawaan). Tidak ada yang terhapus sampai kamu memberi
     `--yakin`.

Dua tahap:
  label ulang : hitung ulang `keywords_matched` dari judul + isi artikel.
                Perlu karena penarikan GDELT lama memberi SEMUA kata kunci pada
                artikel yang judulnya tak memuat satu pun — berita nyasar jadi
                terhitung sebagai topik yang dipantau.
  hapus       : buang dokumen yang tidak memuat kata kunci yang kamu simpan.

Jalankan:
    python tools/bersihkan_db.py                       # pratinjau saja
    python tools/bersihkan_db.py --yakin               # kerjakan
    python tools/bersihkan_db.py --simpan Iran "Selat Hormuz" --yakin
    python tools/bersihkan_db.py --label-ulang-saja --yakin
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cadangkan(db_path: str) -> str:
    cap = datetime.now().strftime("%Y%m%d-%H%M%S")
    tujuan = f"{db_path}.cadangan-{cap}"
    shutil.copy2(db_path, tujuan)
    print(f"Cadangan dibuat: {os.path.basename(tujuan)}")
    return tujuan


def _semua_kata(conn) -> list:
    kata = set()
    for (k,) in conn.execute("SELECT keywords_matched FROM documents"):
        try:
            kata.update(json.loads(k or "[]"))
        except Exception:
            pass
    return sorted(kata)


def label_ulang(conn, kata_kunci: list) -> dict:
    """Hitung ulang keywords_matched dari teks sebenarnya. -> {berubah, kosong}."""
    from collectors.base import match_keywords
    berubah = kosong = 0
    ubah = []
    for r in conn.execute("SELECT doc_id, title, content, keywords_matched "
                          "FROM documents").fetchall():
        lama = sorted(json.loads(r[3] or "[]"))
        baru = sorted(match_keywords(f"{r[1] or ''} {r[2] or ''}", kata_kunci))
        if baru != lama:
            ubah.append((json.dumps(baru, ensure_ascii=False), r[0]))
            berubah += 1
        if not baru:
            kosong += 1
    conn.executemany("UPDATE documents SET keywords_matched=? WHERE doc_id=?", ubah)
    conn.commit()
    return {"berubah": berubah, "tanpa_kata_kunci": kosong}


def bersihkan(db_path: str = "", simpan: list = None, yakin: bool = False,
              label_dulu: bool = True, hapus: bool = True) -> dict:
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"),
                              encoding="utf-8")) or {}
    db_path = db_path or os.path.join(PROJECT_DIR, cfg["storage"]["db_path"])
    simpan = [k for k in (simpan or cfg.get("keywords", [])) if str(k).strip()]
    if not simpan:
        raise SystemExit("Tidak ada kata kunci yang disimpan. Pakai --simpan.")

    conn = sqlite3.connect(db_path, timeout=30)
    # Penarikan arsip bisa sedang menulis ke database yang sama; beri waktu
    # tunggu agar keduanya tidak saling menjatuhkan dengan "database is locked".
    conn.execute("PRAGMA busy_timeout = 30000")
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"Database   : {db_path} ({total} dokumen)")
    print(f"Disimpan   : {', '.join(simpan)}")

    if yakin:
        _cadangkan(db_path)

    if label_dulu:
        # Label ulang memakai GABUNGAN kata kunci lama & baru, supaya dokumen
        # topik lama tetap dikenali topiknya (bukan jadi tanpa label).
        semua = sorted(set(_semua_kata(conn)) | set(simpan))
        if yakin:
            hasil = label_ulang(conn, semua)
            print(f"Label ulang: {hasil['berubah']} dokumen berubah, "
                  f"{hasil['tanpa_kata_kunci']} tak memuat kata kunci apa pun")
        else:
            from collectors.base import match_keywords
            berubah = sum(
                1 for r in conn.execute("SELECT title, content, keywords_matched "
                                        "FROM documents")
                if sorted(match_keywords(f"{r[0] or ''} {r[1] or ''}", semua))
                != sorted(json.loads(r[2] or "[]")))
            print(f"Label ulang: {berubah} dokumen akan berubah labelnya")

    # Dokumen yang DIPERTAHANKAN: teksnya memuat salah satu kata kunci simpan.
    from collectors.base import match_keywords
    buang, tinggal, per_sumber = [], 0, Counter()
    for r in conn.execute("SELECT doc_id, title, content, keywords_matched "
                          "FROM documents").fetchall():
        if match_keywords(f"{r[1] or ''} {r[2] or ''}", simpan):
            tinggal += 1
        else:
            buang.append(r[0])
            for k in json.loads(r[3] or "[]") or ["(tanpa kata kunci)"]:
                per_sumber[k] += 1

    print(f"\nAkan DIHAPUS : {len(buang)} dokumen")
    print(f"Akan DISIMPAN: {tinggal} dokumen")
    if per_sumber:
        print("Rincian yang dihapus (menurut label lamanya):")
        for k, n in per_sumber.most_common(12):
            print(f"  {k:<28} {n}")

    if not hapus:
        print("\n(--label-ulang-saja: tidak ada yang dihapus)")
    elif not yakin:
        print("\nINI BARU PRATINJAU. Tambahkan --yakin untuk benar-benar menghapus.")
    elif buang:
        conn.executemany("DELETE FROM documents WHERE doc_id=?",
                         [(d,) for d in buang])
        # Edge interaksi milik dokumen terhapus ikut dibuang agar SNA tak pincang.
        try:
            conn.execute("DELETE FROM interactions WHERE doc_id NOT IN "
                         "(SELECT doc_id FROM documents)")
        except sqlite3.Error:
            pass
        conn.commit()
        try:                       # VACUUM butuh kunci eksklusif; lewati bila
            conn.execute("VACUUM")  # ada proses lain yang sedang menulis
        except sqlite3.Error as e:
            print(f"  (VACUUM dilewati: {str(e)[:60]} — ukuran berkas menyusut nanti)")
        sisa = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"\nSelesai. Dokumen tersisa: {sisa}")
    conn.close()
    return {"dihapus": len(buang) if yakin and hapus else 0,
            "akan_dihapus": len(buang), "tersisa": tinggal}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = sys.argv[1:]
    simpan = []
    if "--simpan" in a:
        for v in a[a.index("--simpan") + 1:]:
            if v.startswith("--"):
                break
            simpan.append(v)
    bersihkan(simpan=simpan, yakin="--yakin" in a,
              hapus="--label-ulang-saja" not in a)
