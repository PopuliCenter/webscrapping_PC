"""Tarik berita LAMA berdasarkan kata kunci & rentang tanggal (arsip GDELT).

RSS hanya memuat berita terbaru — tidak bisa mundur ke belakang. Untuk menarik
berita yang sudah lewat, dipakai arsip GDELT yang bisa dibatasi per rentang
waktu. GDELT membatasi 250 artikel per permintaan, jadi rentangnya dipecah
per jendela (bawaan: 1 hari) supaya berita tidak terpotong diam-diam.

GDELT hanya memberi JUDUL. Isi artikel diambil menyusul dari situs aslinya
(trafilatura). Untuk berita lama, sebagian gagal diambil — sudah dihapus,
pindah alamat, atau berbayar. Yang gagal tetap disimpan dengan judul saja.

Jalankan:
    python tools/tarik_arsip.py --mulai 2026-08-01 --akhir 2026-09-20
    python tools/tarik_arsip.py --kata "banjir jakarta" "ikn" --mulai 2026-01-01 \
        --akhir 2026-03-31 --jendela 3 --tanpa-isi
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JEDA_ANTAR_JENDELA = 6.0      # detik; GDELT gampang membalas 429 bila diburu


def _tanggal(teks: str) -> datetime:
    return datetime.strptime(teks.strip(), "%Y-%m-%d")


def _jendela(mulai: datetime, akhir: datetime, hari: int):
    """Pecah rentang jadi potongan `hari` hari -> [(mulai, akhir), ...]."""
    potong = []
    a = mulai
    while a < akhir:
        b = min(a + timedelta(days=hari), akhir)
        potong.append((a, b))
        a = b
    return potong


def _kunci_jalan(path: str, paksa: bool = False, kedaluwarsa_jam: float = 6.0):
    """Cegah dua penarikan berjalan bersamaan (mis. tombol tertekan dua kali).

    Dua proses menembak GDELT sekaligus justru saling memicu 429 dan membuat
    keduanya lebih lambat. Kunci dianggap basi setelah beberapa jam supaya
    proses yang mati mendadak tidak memblokir selamanya.
    """
    import json
    if os.path.isfile(path) and not paksa:
        try:
            isi = json.load(open(path, encoding="utf-8"))
            umur = (time.time() - float(isi.get("mulai", 0))) / 3600
        except Exception:
            umur = 999
        if umur < kedaluwarsa_jam:
            raise SystemExit(
                f"Penarikan arsip lain sedang berjalan (mulai {umur * 60:.0f} menit lalu, "
                f"PID {isi.get('pid', '?')}).\n"
                f"Tunggu sampai selesai, atau pakai --paksa bila yakin sudah mati.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "mulai": time.time()}, f)


def _isi_penuh(docs: list, pekerja: int = 12, kata_kunci: list = None) -> int:
    """Ambil isi artikel dari situs aslinya, paralel secukupnya. -> jumlah berhasil."""
    from collectors.news_rss import _full_text
    if not docs:
        return 0

    from collectors.base import match_keywords

    def satu(d):
        teks = _full_text(d.url)
        berhasil = 0
        if teks and len(teks) > len(d.content or ""):
            d.content = teks
            berhasil = 1
        # Label ulang setelah isi artikel ada: GDELT hanya memberi judul, jadi
        # kata kunci yang muncul di badan berita baru ketahuan di sini.
        if kata_kunci:
            d.keywords_matched = match_keywords(f"{d.title} {d.content}", kata_kunci)
        return berhasil

    with ThreadPoolExecutor(max_workers=pekerja) as ex:
        return sum(ex.map(satu, docs))


def tarik(kata: list, mulai: str, akhir: str, lang: str = "id", jendela: int = 1,
          maks: int = 250, isi_penuh: bool = True, db_path: str = "",
          analisis: bool = True, pekerja: int = 12, paksa: bool = False) -> dict:
    import yaml
    from collectors import news_gdelt
    from core.storage import Storage

    cfg_path = os.path.join(PROJECT_DIR, "config.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    kata = [k for k in (kata or cfg.get("keywords", [])) if str(k).strip()]
    if not kata:
        raise SystemExit("Tidak ada kata kunci. Pakai --kata atau isi keywords di config.yaml")
    db_path = db_path or os.path.join(PROJECT_DIR, cfg["storage"]["db_path"])

    d1, d2 = _tanggal(mulai), _tanggal(akhir) + timedelta(days=1)   # akhir inklusif
    if d1 >= d2:
        raise SystemExit("Tanggal mulai harus sebelum tanggal akhir.")
    potong = _jendela(d1, d2, max(1, jendela))
    _kunci_jalan(os.path.join(PROJECT_DIR, "data", ".arsip_berjalan.json"), paksa)
    store = Storage(db_path)

    print(f"Kata kunci : {', '.join(kata)}")
    print(f"Rentang    : {mulai} s/d {akhir}  ({len(potong)} jendela x {jendela} hari)")
    print(f"Isi artikel: {'ya, ' + str(pekerja) + ' paralel' if isi_penuh else 'tidak (judul saja)'}")
    t_mulai = time.time()

    hitung = {"artikel": 0, "baru": 0, "isi": 0}

    def satu_jendela(a, b, label: str) -> bool:
        """-> True bila permintaan berhasil (walau nol artikel)."""
        t0 = time.time()
        try:
            docs = news_gdelt.collect(kata, lang, max_records=maks,
                                      mulai=a.strftime("%Y%m%d%H%M%S"),
                                      akhir=b.strftime("%Y%m%d%H%M%S"),
                                      percobaan=4, jeda=8, ketat=True)
        except news_gdelt.GdeltGagal as e:
            print(f"  {label} {a:%Y-%m-%d}: GAGAL ({e}) — akan diulang", flush=True)
            return False
        t_gdelt = time.time() - t0
        t1 = time.time()
        n_isi = _isi_penuh(docs, pekerja, kata) if (isi_penuh and docs) else 0
        t_isi = time.time() - t1
        baru = store.save_documents(docs)
        hitung["artikel"] += len(docs)
        hitung["baru"] += baru
        hitung["isi"] += n_isi
        tanda = "  <- mungkin terpotong, kecilkan --jendela" if len(docs) >= maks else ""
        print(f"  {label} {a:%Y-%m-%d}: {len(docs):>3} artikel, {baru:>3} baru, "
              f"isi penuh {n_isi} ({t_gdelt:.0f}s cari + {t_isi:.0f}s isi){tanda}",
              flush=True)
        return True

    gagal = []
    for i, (a, b) in enumerate(potong, 1):
        if not satu_jendela(a, b, f"[{i:>3}/{len(potong)}]"):
            gagal.append((a, b))
        if i < len(potong):
            time.sleep(JEDA_ANTAR_JENDELA)

    # Putaran ulang: jendela gagal ditarik lagi dengan jeda lebih longgar.
    # Tanpa ini, hari yang gagal terlihat sama dengan hari yang memang sepi.
    if gagal:
        print(f"\nMengulang {len(gagal)} jendela yang gagal (jeda lebih longgar)...")
        masih = []
        for i, (a, b) in enumerate(gagal, 1):
            time.sleep(JEDA_ANTAR_JENDELA * 3)
            if not satu_jendela(a, b, f"[ulang {i}/{len(gagal)}]"):
                masih.append((a, b))
        gagal = masih

    menit = (time.time() - t_mulai) / 60
    print(f"\nTotal: {hitung['artikel']} artikel, {hitung['baru']} baru disimpan, "
          f"{hitung['isi']} berisi teks penuh — {menit:.1f} menit")
    if gagal:
        tanggal = ", ".join(f"{a:%Y-%m-%d}" for a, _ in gagal[:8])
        print(f"PERINGATAN: {len(gagal)} jendela tetap gagal ({tanggal}"
              f"{'...' if len(gagal) > 8 else ''}). Berita pada tanggal itu BELUM "
              f"tertarik — jalankan ulang perintah yang sama nanti; yang sudah "
              f"tersimpan tidak akan terduplikasi.")
    total_art, total_baru, total_isi = hitung["artikel"], hitung["baru"], hitung["isi"]
    # Jalankan walau tak ada yang baru: bisa jadi masih ada dokumen lama yang
    # belum dinilai (mis. penarikan sebelumnya memakai --tanpa-analisis).
    if analisis:
        from run_once import analyze_sentiment
        n = analyze_sentiment(cfg, store)
        print(f"Sentimen dianalisis: {n} dokumen")
    print("Statistik DB:", store.stats())
    try:                                        # lepaskan kunci agar bisa dijalankan lagi
        os.remove(os.path.join(PROJECT_DIR, "data", ".arsip_berjalan.json"))
    except OSError:
        pass
    return {"artikel": total_art, "baru": total_baru, "isi_penuh": total_isi,
            "jendela_gagal": [a.strftime("%Y-%m-%d") for a, _ in gagal]}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = sys.argv[1:]

    def opsi(nama, bawaan=None):
        return a[a.index(nama) + 1] if nama in a else bawaan

    kata = []
    if "--kata" in a:                      # semua nilai sampai flag berikutnya
        for v in a[a.index("--kata") + 1:]:
            if v.startswith("--"):
                break
            kata.append(v)
    kemarin = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    tarik(kata,
          mulai=opsi("--mulai", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")),
          akhir=opsi("--akhir", kemarin),
          lang=opsi("--lang", "id"),
          jendela=int(opsi("--jendela", 1)),
          maks=int(opsi("--maks", 250)),
          isi_penuh="--tanpa-isi" not in a,
          analisis="--tanpa-analisis" not in a,
          pekerja=int(opsi("--pekerja", 12)),
          paksa="--paksa" in a)
