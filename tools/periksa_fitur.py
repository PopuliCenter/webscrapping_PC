"""Periksa semua fitur analisis dengan MENJALANKANNYA pada data nyata.

Bukan sekadar mengecek modul bisa diimpor: tiap fitur benar-benar dijalankan
atas isi database, lalu hasilnya ditampilkan supaya bisa dinilai sendiri —
"jalan" berbeda dari "hasilnya masuk akal".

Fitur yang butuh data media sosial (SNA, bot/buzzer) akan dilaporkan sebagai
DILEWATI bila database hanya berisi berita; itu bukan kegagalan, melainkan
konsekuensi sumber datanya.

Jalankan:
    python tools/periksa_fitur.py              # semua kecuali yang berat
    python tools/periksa_fitur.py --berat      # termasuk zero-shot & banding model
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HASIL = []


def bagian(nama: str):
    print(f"\n{'─' * 70}\n{nama}\n{'─' * 70}")


def lapor(fitur: str, status: str, catatan: str = ""):
    HASIL.append((fitur, status, catatan))
    tanda = {"OK": "[ OK ]", "LEMAH": "[LEMAH]", "LEWAT": "[LEWAT]", "GAGAL": "[GAGAL]"}
    print(f"  {tanda.get(status, status)} {fitur}" + (f" — {catatan}" if catatan else ""))


def ambil_teks(db_path: str, n: int = 60) -> tuple:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    baris = conn.execute(
        "SELECT title, content, sentiment_label, platform FROM documents "
        "WHERE content IS NOT NULL AND length(content) > 200 LIMIT ?", (n,)).fetchall()
    conn.close()
    teks = [f"{r['title']} {r['content']}" for r in baris]
    label = [r["sentiment_label"] for r in baris]
    return teks, label


def periksa_preprocessing(teks: list):
    bagian("1. PREPROCESSING (emoji, slang, negasi, stemming)")
    from analysis.preprocess import Preprocessor
    p = Preprocessor(use_stemmer=True)
    contoh = ("Gak bgt sih 😡😡 harga BBM naik lagi https://t.co/x @jokowi #protes "
              "Ini TIDAK BAGUS bgt!!! 1000%")
    langkah = p.steps(contoh)
    for k, v in langkah.items():
        cuplik = v if isinstance(v, str) else " ".join(map(str, v))
        print(f"    {k:<14}: {cuplik[:78]}")
    token = p.tokens(contoh)
    lapor("Hapus emoji/URL/mention/hashtag", "OK" if "😡" not in " ".join(token) else "GAGAL")
    # "gak"->"tidak", "bgt"->"sangat"; cek prefiks karena negasi sudah digabung
    baku = any(t.startswith("tidak") or t.startswith("sangat") for t in token)
    lapor("Normalisasi slang (gak/bgt -> baku)", "OK" if baku else "LEMAH",
          " ".join(token[:6]))
    neg = [t for t in token if "_" in t]
    lapor("Gabung negasi (tidak_bagus)", "OK" if neg else "LEMAH", ", ".join(neg[:3]))
    lapor("Stemming Sastrawi", "OK" if p.stemming_active else "GAGAL")
    lapor("Buang angka & tanda baca", "OK" if not any(t.isdigit() for t in token) else "GAGAL")


def periksa_textstats(teks: list):
    bagian("2. TEXT MINING (frekuensi, n-gram, kata khas)")
    from analysis import textstats
    from analysis.preprocess import Preprocessor
    p = Preprocessor(use_stemmer=False)
    freq = textstats.word_freq(teks, p, top_n=8)
    bi = textstats.ngrams(teks, 2, p, top_n=5)
    print(f"    kata teratas : {', '.join(f'{k}({v})' for k, v in freq)}")
    print(f"    bigram       : {', '.join(f'{k}({v})' for k, v in bi)}")
    lapor("Frekuensi kata", "OK" if freq else "GAGAL", f"{len(freq)} kata")
    lapor("N-gram (bigram)", "OK" if bi else "GAGAL")
    try:
        from wordcloud import WordCloud          # noqa: F401
        lapor("Word cloud", "OK", "pustaka tersedia")
    except Exception as e:
        lapor("Word cloud", "GAGAL", str(e)[:40])


def periksa_lexicon(teks: list):
    bagian("3. SKOR PER-KATA (bisa diaudit)")
    from analysis.lexicon_id import LexiconScorer
    s = LexiconScorer()
    hasil = s.score("harga naik terus, rakyat kecewa tapi pemerintah tetap optimis")
    rinci = hasil.get("details") or []
    print(f"    label {hasil['label']} skor {hasil['score']} | {len(rinci)} kata bersentimen")
    for d in rinci[:3]:
        print(f"      {d['kata']:<12} bobot {d['bobot_dasar']:+.2f} x faktor {d['faktor']} = {d['kontribusi']:+.2f} ({d['alasan']})")
    lapor("Penilaian per kata (negasi & penguat)", "OK" if rinci else "LEMAH",
          f"{len(rinci)} kata dirinci")


def periksa_smote(db_path: str):
    bagian("4. KLASIFIKASI ML (TF-IDF + SMOTE, 6 algoritma)")
    from analysis.ml_classify import load_dataset_from_db, run_comparison, summary_table
    teks, label = load_dataset_from_db(db_path)
    print(f"    dataset: {len(teks)} dokumen, distribusi {dict((l, label.count(l)) for l in set(label))}")
    if len(set(label)) < 2 or len(teks) < 40:
        lapor("TF-IDF + SMOTE", "LEWAT", "data terlalu sedikit / satu kelas saja")
        return
    res = run_comparison(teks, label)
    if "error" in res:
        lapor("TF-IDF + SMOTE", "GAGAL", res["error"][:60])
        return
    print(f"    fitur TF-IDF : {res['fitur_tfidf']}")
    print(f"    SMOTE        : {res['smote']}")
    print(f"    distribusi   : {res['distribusi_sebelum_smote']} -> {res['distribusi_sesudah_smote']}")
    for r in summary_table(res)[:6]:
        print(f"      {r['Model']:<22} F1 {r['F1 (before)']} -> {r['F1 (after)']}  (Δ {r['Δ F1']})")
    lapor("TF-IDF", "OK", f"{res['fitur_tfidf']} fitur")
    lapor("SMOTE before/after", "OK" if "berhasil" in res["smote"] else "LEMAH", res["smote"])
    lapor("6 algoritma klasifikasi", "OK" if len(res["hasil"]) == 6 else "LEMAH",
          f"{len(res['hasil'])} model")
    catatan = ("label latih = tebakan IndoBERT, bukan label manusia — "
               "angkanya mengukur kemiripan, bukan kebenaran")
    lapor("Label acuan tab ini", "LEMAH", catatan)


def periksa_sentimen(teks: list):
    bagian("5. SENTIMEN (IndoBERT + per paragraf)")
    from analysis.sentiment import SentimentEngine
    eng = SentimentEngine({"engine": "indobert",
                           "model_dir": "models/indobert-sentiment-finetuned"})
    t0 = time.time()
    hasil = eng.predict_panjang_batch(teks[:20])
    dt = time.time() - t0
    panjang = [h for h in hasil if h["bagian_total"] > 1]
    print(f"    {len(teks[:20])} dokumen dinilai dalam {dt:.1f} dtk "
          f"({len(panjang)} dinilai per paragraf)")
    print(f"    contoh: {hasil[0]['label']} {hasil[0]['score']} "
          f"({hasil[0]['bagian_negatif']}/{hasil[0]['bagian_total']} paragraf negatif)")
    lapor("Model lokal hasil fine-tuning", "OK", os.path.basename(eng.model_source))
    lapor("Penilaian per paragraf", "OK" if panjang else "LEMAH",
          f"{len(panjang)}/20 dokumen")
    lapor("Perangkat", "OK", str(eng._pipe.device))


def periksa_emosi(db_path: str, teks: list):
    bagian("6. EMOSI")
    from analysis.emotion import EmotionEngine
    eng = EmotionEngine()
    hasil = eng.predict_batch(teks[:15]) if hasattr(eng, "predict_batch") else None
    if hasil is None:
        hasil = [eng.predict(t) for t in teks[:15]]
    from collections import Counter
    dist = Counter(h[0] for h in hasil)
    print(f"    distribusi 15 dokumen: {dict(dist)}")
    lapor("Analisis emosi 5 kelas", "OK" if dist else "GAGAL", str(dict(dist))[:50])


def periksa_sarkasme(teks: list):
    bagian("7. SARKASME")
    from analysis.sarcasm import SarcasmEngine
    eng = SarcasmEngine()
    contoh = ["wah hebat, harga naik terus. prestasi luar biasa",
              "harga cabai turun di pasar induk"]
    hasil = eng.predict(contoh)
    for t, (l, p) in zip(contoh, hasil):
        print(f"    {l:<13} p={p:.3f}  <- {t[:50]}")
    lapor("Deteksi sarkasme", "OK" if hasil[0][0] != hasil[1][0] else "LEMAH",
          os.path.basename(eng.sumber))


def periksa_sosial(db_path: str):
    bagian("8. MEDIA SOSIAL (X, Instagram, SNA, bot/buzzer)")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"), encoding="utf-8"))
    conn = sqlite3.connect(db_path)
    per_plat = dict(conn.execute(
        "SELECT platform, COUNT(*) FROM documents GROUP BY platform").fetchall())
    n_edge = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    conn.close()
    print(f"    dokumen per platform: {per_plat} | edge interaksi: {n_edge}")

    for nama, kunci, berkas in (("X/Twitter", "x", "secrets/x_accounts.yaml"),
                                ("Instagram", "instagram", "secrets/ig_accounts.yaml")):
        aktif = cfg.get(kunci, {}).get("enabled")
        ada_akun = os.path.isfile(os.path.join(PROJECT_DIR, berkas))
        if not ada_akun:
            lapor(f"{nama} — kredensial", "LEWAT", f"{berkas} belum ada")
        else:
            lapor(f"{nama} — kredensial", "OK", "terisi")
        lapor(f"{nama} — aktif di config", "OK" if aktif else "LEWAT",
              "enabled: true" if aktif else "enabled: false")

    for pustaka in ("twikit", "instaloader"):
        try:
            __import__(pustaka)
            lapor(f"Pustaka {pustaka}", "OK", "terpasang")
        except Exception:
            lapor(f"Pustaka {pustaka}", "GAGAL", "belum terpasang")

    from analysis import sna
    G = sna.build_graph(db_path)
    lapor("SNA (graf interaksi)", "OK" if G.number_of_nodes() else "LEWAT",
          f"{G.number_of_nodes()} node, {G.number_of_edges()} edge — "
          f"butuh data X/IG" if not G.number_of_nodes() else
          f"{G.number_of_nodes()} node")
    from analysis import bot_detect
    aktor = bot_detect.analyze_actors(db_path, platform="twitter")
    lapor("Deteksi bot/buzzer", "OK" if aktor else "LEWAT",
          f"{len(aktor)} aktor — butuh data X/IG" if not aktor else f"{len(aktor)} aktor")


def periksa_berat(teks: list):
    bagian("9. FITUR BERAT (zero-shot intent, intensitas 5 tingkat)")
    from analysis.zeroshot import IntentZeroShot, IntensitasPolaritas
    zs = IntentZeroShot()
    hasil = zs.predict(teks[:3]) if hasattr(zs, "predict") else None
    print(f"    intent 3 dokumen: {hasil}")
    lapor("Intent zero-shot", "OK" if hasil else "LEMAH")
    ip = IntensitasPolaritas()
    tingkat = [ip.predict(t)[0] for t in teks[:3]] if hasattr(ip, "predict") else None
    print(f"    intensitas: {tingkat}")
    lapor("Intensitas 5 tingkat", "OK" if tingkat else "LEMAH")


def main(berat: bool = False):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"), encoding="utf-8"))
    db_path = os.path.join(PROJECT_DIR, cfg["storage"]["db_path"])
    teks, label = ambil_teks(db_path)
    print(f"Database: {db_path}")
    print(f"Contoh uji: {len(teks)} dokumen berisi teks panjang")

    urut = [(periksa_preprocessing, (teks,)), (periksa_textstats, (teks,)),
            (periksa_lexicon, (teks,)), (periksa_smote, (db_path,)),
            (periksa_sentimen, (teks,)), (periksa_emosi, (db_path, teks)),
            (periksa_sarkasme, (teks,)), (periksa_sosial, (db_path,))]
    if berat:
        urut.append((periksa_berat, (teks,)))
    for fn, arg in urut:
        try:
            fn(*arg)
        except Exception as e:
            lapor(fn.__name__, "GAGAL", f"{type(e).__name__}: {str(e)[:70]}")

    bagian("RINGKASAN")
    from collections import Counter
    c = Counter(s for _, s, _ in HASIL)
    for fitur, status, catatan in HASIL:
        if status != "OK":
            print(f"  {status:<6} {fitur} — {catatan}")
    print(f"\n  OK {c['OK']} | LEMAH {c['LEMAH']} | DILEWATI {c['LEWAT']} | GAGAL {c['GAGAL']}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    main(berat="--berat" in sys.argv)
