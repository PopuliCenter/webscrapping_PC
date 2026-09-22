"""Share of Voice — porsi pemberitaan tiap pihak/isu, beserta nadanya.

Pertanyaan khas media specialist: "dari semua pemberitaan pekan ini, berapa
persen membicarakan A dibanding B, dan mana yang nadanya lebih negatif?"

Tiga aturan yang dipakai supaya angkanya tidak menipu:

  1. Dihitung per DOKUMEN, bukan per kemunculan kata. Satu artikel yang menyebut
     satu nama 30 kali tetap dihitung satu.
  2. Berita sindikasi (kolom `duplikat_dari`) dihitung SEKALI. Satu rilis yang
     dimuat 10 media bukan berarti 10 kali pemberitaan.
  3. Satu artikel boleh masuk ke beberapa pihak sekaligus bila memang
     menyebut semuanya — maka jumlah persentase bisa melebihi 100%, dan itu
     dinyatakan terang-terangan, bukan dipaksa jadi 100%.

Indeks nada: (positif - negatif) / jumlah dokumen, rentang -1..+1.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from collectors.base import match_keywords


def muat_dokumen(db_path: str, mulai: str = "", akhir: str = "",
                 abaikan_duplikat: bool = True) -> pd.DataFrame:
    """Ambil dokumen dari database, siap dihitung."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM documents", conn)
    conn.close()
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["dt"] = df["dt"].fillna(pd.to_datetime(df["collected_at"], errors="coerce", utc=True))
    df["teks"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    if abaikan_duplikat and "duplikat_dari" in df.columns:
        df = df[df["duplikat_dari"].fillna("").astype(str).str.len() == 0]
    if mulai:
        df = df[df["dt"] >= pd.Timestamp(mulai, tz="UTC")]
    if akhir:
        df = df[df["dt"] < pd.Timestamp(akhir, tz="UTC") + pd.Timedelta(days=1)]
    return df


def hitung(df: pd.DataFrame, entitas: list, pakai_bobot: bool = True) -> pd.DataFrame:
    """-> tabel per entitas: jumlah, share %, sebaran sentimen, indeks nada.

    `pakai_bobot=True` menambah kolom share BERBOBOT: satu berita di media
    nasional besar dihitung lebih berat daripada di blog daerah (lihat
    analysis/media.py). Share mentah tetap ditampilkan berdampingan — keduanya
    menjawab pertanyaan berbeda: "seberapa sering" vs "seberapa besar gaungnya".
    """
    if df.empty or not entitas:
        return pd.DataFrame()
    total = len(df)
    if pakai_bobot:
        from .media import tambah_bobot
        df = tambah_bobot(df)
        total_bobot = float(df["bobot"].sum()) or 1.0
    baris = []
    for e in entitas:
        kena = df[df["teks"].map(lambda t: bool(match_keywords(t, [e])))]
        n = len(kena)
        if not n:
            kosong = {"entitas": e, "dokumen": 0, "share_%": 0.0, "positif": 0,
                      "netral": 0, "negatif": 0, "indeks_nada": 0.0,
                      "media": 0, "media_teratas": ""}
            if pakai_bobot:
                kosong["share_bobot_%"] = 0.0
            baris.append(kosong)
            continue
        s = kena["sentiment_label"].value_counts()
        pos, net, neg = int(s.get("positive", 0)), int(s.get("neutral", 0)), int(s.get("negative", 0))
        media = kena["source"].value_counts()
        baris.append({
            "entitas": e,
            "dokumen": n,
            "share_%": round(100 * n / total, 1),
            **({"share_bobot_%": round(100 * float(kena["bobot"].sum()) / total_bobot, 1)}
               if pakai_bobot else {}),
            "positif": pos, "netral": net, "negatif": neg,
            "indeks_nada": round((pos - neg) / n, 3),
            "media": int(kena["source"].nunique()),
            "media_teratas": ", ".join(f"{k} ({v})" for k, v in media.head(3).items()),
        })
    tabel = pd.DataFrame(baris).sort_values("dokumen", ascending=False)
    return tabel.reset_index(drop=True)


def tren(df: pd.DataFrame, entitas: list, freq: str = "D") -> pd.DataFrame:
    """Jumlah dokumen per entitas per hari -> untuk grafik tren share."""
    if df.empty or not entitas:
        return pd.DataFrame()
    d = df.dropna(subset=["dt"]).copy()
    if d.empty:
        return pd.DataFrame()
    # buang zona waktu dulu: to_period() memang tidak menyimpannya
    lokal = d["dt"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    d["tanggal"] = lokal.dt.to_period(freq).dt.to_timestamp()
    keluar = {}
    for e in entitas:
        kena = d[d["teks"].map(lambda t: bool(match_keywords(t, [e])))]
        keluar[e] = kena.groupby("tanggal").size()
    return pd.DataFrame(keluar).fillna(0).astype(int)


def lonjakan(df: pd.DataFrame, ambang: float = 2.0, dasar_min: float = 5.0) -> Optional[dict]:
    """Deteksi lonjakan volume hari terakhir dibanding rata-rata sebelumnya.

    Dipakai sebagai peringatan dini: isu yang tiba-tiba ramai perlu respons
    cepat. Dua pengaman terhadap alarm palsu:
      - butuh minimal 4 hari data;
      - rata-rata hari sebelumnya harus >= `dasar_min`. Tanpa ini, pemantauan
        yang baru dimulai selalu tampak "melonjak" — kemarin nyaris nol bukan
        karena isunya sepi, tapi karena datanya memang belum terkumpul.
    """
    if df.empty or df["dt"].isna().all():
        return None
    harian = (df.dropna(subset=["dt"])
              .set_index("dt").tz_convert("Asia/Jakarta")
              .resample("D").size())
    if len(harian) < 4:
        return None
    terakhir, sebelumnya = harian.iloc[-1], harian.iloc[:-1]
    rata = sebelumnya.mean()
    if rata <= 0:
        return None
    rasio = terakhir / rata
    cukup = rata >= dasar_min
    return {"tanggal": str(harian.index[-1].date()), "jumlah": int(terakhir),
            "rata_rata": round(float(rata), 1), "rasio": round(float(rasio), 2),
            "dasar_cukup": bool(cukup),
            "lonjakan": bool(cukup and rasio >= ambang),
            "catatan": "" if cukup else
                       f"riwayat terlalu tipis (rata-rata {rata:.1f}/hari) — "
                       f"belum bisa dipakai sebagai peringatan"}

def saring_relevan(df: pd.DataFrame, entitas: list, min_sebut: int = 2) -> tuple:
    """Buang berita yang hanya MENYINGGUNG kata kunci sekilas.

    Kata kunci bisa muncul di berita yang sama sekali bukan topikmu: "dolar AS"
    di berita judi, atau inisial nama orang. Aturan yang dipakai — berita
    dianggap relevan bila kata kuncinya ada di JUDUL, atau disebut minimal
    `min_sebut` kali di seluruh teks.

    -> (df_relevan, jumlah_dibuang)
    """
    import re
    if df.empty or not entitas:
        return df, 0

    def relevan(baris) -> bool:
        judul = str(baris.get("title") or "")
        teks = str(baris.get("teks") or "")
        if match_keywords(judul, entitas):
            return True
        for e in entitas:
            pola = _pola_hitung(e)
            if len(pola.findall(teks)) >= min_sebut:
                return True
        return False

    tanda = df.apply(relevan, axis=1)
    return df[tanda], int((~tanda).sum())


_cache_pola: dict = {}


def _pola_hitung(kata: str):
    """Pola untuk MENGHITUNG kemunculan; ikut aturan huruf besar match_keywords."""
    import re
    from collectors.base import singkatan_kapital
    pola = _cache_pola.get(kata)
    if pola is None:
        k = kata.strip()
        bendera = 0 if singkatan_kapital(k) else re.IGNORECASE
        pola = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", bendera)
        _cache_pola[kata] = pola
    return pola
