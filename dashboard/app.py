"""Dashboard monitoring (Streamlit).

Tab:
  Ringkasan · Topik · Teks & Word Cloud · Heatmap · Jaringan (SNA) ·
  Bot/Buzzer · Klasifikasi ML · Emosi · Intent & Sarkasme · Banding Model

Jalankan:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import sna                                  # noqa: E402
from analysis.preprocess import Preprocessor              # noqa: E402
from analysis import textstats                            # noqa: E402

st.set_page_config(page_title="Monitoring Sosial", layout="wide")

_SENT_VAL = {"positive": 1, "neutral": 0, "negative": -1}
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


# ── Data ────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _cfg():
    return yaml.safe_load(open("config.yaml", encoding="utf-8"))


def _db_path():
    return _cfg()["storage"]["db_path"]


@st.cache_data(ttl=60)
def load_docs() -> pd.DataFrame:
    conn = sqlite3.connect(_db_path())
    df = pd.read_sql_query("SELECT * FROM documents", conn)
    conn.close()
    if not df.empty:
        df["dt"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["dt"] = df["dt"].fillna(pd.to_datetime(df["collected_at"], errors="coerce", utc=True))
        df["teks"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    return df


@st.cache_resource
def get_preprocessor(stem: bool = True) -> Preprocessor:
    return Preprocessor(use_stemmer=stem)


@st.cache_data(ttl=300)
def cached_freq(texts: tuple, stem: bool, per_dokumen: bool = False) -> dict:
    return textstats.freq_dict(list(texts), get_preprocessor(stem), per_dokumen)


# ── Panel kata kunci ────────────────────────────────────────────
# Sumbernya beda gaya penulisan: berita pakai frasa ("ibu kota nusantara"),
# X pakai query pendek, IG pakai tagar tanpa spasi. Karena itu dipisah.
SUMBER_KATA_KUNCI = [
    ("Berita (RSS + GDELT)", ["keywords"],
     "Satu per baris. Berita yang tidak memuat salah satunya dibuang; "
     "dipakai juga sebagai query GDELT."),
    ("X / Twitter", ["x", "search_queries"],
     "Query pencarian X. Biasanya kata pendek, boleh beda dari kata kunci berita."),
    ("Instagram", ["instagram", "hashtags"],
     "Tagar tanpa tanda # dan tanpa spasi, mis. ibukotanusantara."),
]


# Rem pengaman yang boleh disetel dari dashboard. Format:
#   (judul, path di config.yaml, satuan, langkah, bantuan)
SETELAN_REM = {
    "X / Twitter": [
        ("Batas per hari", ["x", "daily_limit"], "tweet", 50,
         "Total tweet per HARI, lintas query & siklus. 0 = tanpa batas."),
        ("Ambil per query", ["x", "tweets_per_query"], "tweet", 10,
         "Berapa tweet diambil tiap query dalam satu siklus."),
        ("Jeda minimum", ["x", "twikit", "min_delay_sec"], "detik", 1,
         "Jeda acak antar permintaan; makin besar makin aman."),
        ("Jeda maksimum", ["x", "twikit", "max_delay_sec"], "detik", 1, ""),
        ("Istirahat akun", ["x", "twikit", "cooldown_minutes"], "menit", 5,
         "Lama akun diistirahatkan setelah kena limit."),
    ],
    "Instagram": [
        ("Batas per hari", ["instagram", "daily_limit"], "post", 25,
         "Total post per HARI. 0 = tanpa batas."),
        ("Ambil per tagar", ["instagram", "posts_per_tag"], "post", 5, ""),
        ("Jeda minimum", ["instagram", "min_delay_sec"], "detik", 1,
         "IG jauh lebih agresif — jeda besar sangat disarankan."),
        ("Jeda maksimum", ["instagram", "max_delay_sec"], "detik", 1, ""),
        ("Istirahat akun", ["instagram", "cooldown_minutes"], "menit", 15, ""),
    ],
}


def panel_arsip():
    """Tarik berita LAMA per kata kunci & rentang tanggal (arsip GDELT)."""
    import datetime as _dt
    from core.config_edit import baca_list

    proyek = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(proyek, "config.yaml")
    with st.expander("🗓️ Tarik berita lama (arsip)", expanded=False):
        st.caption("RSS hanya memuat berita terbaru. Untuk berita yang sudah lewat, "
                   "dipakai arsip GDELT — dibatasi kata kunci dan rentang tanggal.")
        # Kunci widget ikut isi config: begitu config.yaml berubah (mis. lewat
        # panel Kata kunci), kotak ini ikut berubah. Tanpa itu Streamlit menahan
        # teks lama di session dan kamu menarik arsip dengan kata kunci usang.
        kata_cfg = baca_list(cfg_path, ["keywords"])
        kata = st.text_area("Kata kunci (satu per baris)", "\n".join(kata_cfg),
                            height=80, key=f"arsip_kata_{hash(tuple(kata_cfg))}")
        hari_ini = _dt.date.today()
        k1, k2 = st.columns(2)
        d_mulai = k1.date_input("Dari", hari_ini - _dt.timedelta(days=14),
                                max_value=hari_ini, key="arsip_mulai")
        d_akhir = k2.date_input("Sampai", hari_ini - _dt.timedelta(days=1),
                                max_value=hari_ini, key="arsip_akhir")
        isi_penuh = st.checkbox("Ambil isi artikel penuh", value=True,
                                help="Lebih lambat. Untuk berita lama sebagian gagal "
                                     "karena halamannya sudah dihapus atau berbayar.",
                                key="arsip_isi")
        hari = (d_akhir - d_mulai).days + 1
        if hari > 0:
            # Perkiraan diukur dari pemakaian nyata: mengambil isi artikel jauh
            # lebih mahal daripada mencarinya (±2,5 dtk/artikel dibagi jumlah
            # pekerja). Ditulis sebagai RENTANG karena jumlah artikel per hari
            # tak diketahui sebelum ditarik, dan GDELT kadang membatasi (429).
            if isi_penuh:
                st.caption(f"Perkiraan **{hari * 1:.0f}–{hari * 3:.0f} menit** "
                           f"({hari} hari). Sebagian besar waktu habis untuk "
                           f"mengambil isi artikel — makin banyak berita yang "
                           f"cocok, makin lama.")
            else:
                st.caption(f"Perkiraan **{max(1, hari * 10 // 60)}–{max(2, hari * 20 // 60)} menit** "
                           f"({hari} hari, judul saja). Jauh lebih cepat, tapi "
                           f"sentimen dari judul saja lebih lemah.")
        # Kunci dari proses yang sedang jalan: tombol dimatikan supaya tidak
        # ada dua penarikan bersamaan (keduanya jadi lambat karena 429).
        kunci = os.path.join(proyek, "data", ".arsip_berjalan.json")
        sedang_jalan = os.path.isfile(kunci)
        if sedang_jalan:
            st.warning("Ada penarikan arsip yang sedang berjalan. Tunggu sampai "
                       "selesai — menjalankan dua sekaligus justru memperlambat "
                       "keduanya.")
        if st.button("⬇️ Tarik arsip", width="stretch", key="arsip_jalan",
                     disabled=sedang_jalan):
            daftar = [b.strip() for b in kata.splitlines() if b.strip()]
            if not daftar:
                st.error("Isi kata kunci dulu.")
            elif d_mulai > d_akhir:
                st.error("Tanggal 'Dari' harus sebelum 'Sampai'.")
            else:
                import re
                import subprocess
                perintah = [sys.executable, "-u", "tools/tarik_arsip.py",
                            "--kata", *daftar, "--mulai", d_mulai.isoformat(),
                            "--akhir", d_akhir.isoformat()]
                if not isi_penuh:
                    perintah.append("--tanpa-isi")

                bar = st.progress(0.0)
                status = st.empty()
                layar = st.empty()
                keluaran = []
                # Dibaca baris demi baris supaya progres terlihat sejak awal —
                # menunggu proses selesai tanpa kabar apa pun membuat penarikan
                # panjang terasa menggantung.
                p = subprocess.Popen(perintah, cwd=proyek, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1)
                pola = re.compile(r"\[\s*(\d+)/(\d+)\]")
                for baris in p.stdout:
                    baris = baris.rstrip()
                    if not baris or "it/s]" in baris:
                        continue
                    keluaran.append(baris)
                    m = pola.search(baris)
                    if m:
                        bar.progress(min(1.0, int(m.group(1)) / int(m.group(2))))
                    status.caption(baris[:120])
                    layar.code("\n".join(keluaran[-12:]))
                p.wait()
                bar.progress(1.0)
                status.empty()
                layar.code("\n".join(keluaran[-20:]) or "(tanpa keluaran)")
                if p.returncode == 0:
                    load_docs.clear()
                    st.success("Selesai. Muat ulang halaman untuk melihat datanya.")
                else:
                    st.error("Gagal — lihat pesan di atas.")


def panel_rem():
    """Sunting batas harian, jeda & istirahat akun -> config.yaml."""
    from core.config_edit import baca_scalar, set_scalar

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")
    with st.expander("🛡️ Batas & jeda (anti-banned)", expanded=False):
        st.caption("Rem pengaman penarikan medsos. Makin kecil & makin lambat, "
                   "makin kecil risiko akun dibatasi.")
        with st.form("form_rem"):
            isian = {}
            for judul, baris in SETELAN_REM.items():
                st.markdown(f"**{judul}**")
                for label, path, satuan, langkah, bantuan in baris:
                    isian[tuple(path)] = st.number_input(
                        f"{label} ({satuan})", min_value=0, step=langkah,
                        value=int(baca_scalar(cfg_path, path, 0) or 0),
                        help=bantuan or None, key="rem_" + "_".join(path))
            simpan = st.form_submit_button("Simpan", width="stretch")

        if simpan:
            salah = []
            for judul, baris in SETELAN_REM.items():                # jeda min <= maks
                jeda = {l: isian[tuple(p)] for l, p, *_ in baris if "Jeda" in l}
                if jeda.get("Jeda minimum", 0) > jeda.get("Jeda maksimum", 0):
                    salah.append(f"{judul}: jeda minimum melebihi maksimum")
            if salah:
                st.error(" · ".join(salah) + " — tidak ada yang disimpan.")
                return
            ubah = []
            for judul, baris in SETELAN_REM.items():
                for label, path, satuan, *_ in baris:
                    baru = int(isian[tuple(path)])
                    if baru != baca_scalar(cfg_path, path, 0):
                        try:
                            set_scalar(cfg_path, path, baru)
                            ubah.append(f"{judul} · {label} = {baru} {satuan}")
                        except Exception as e:
                            st.error(f"{judul} · {label} gagal disimpan: {e}")
            if ubah:
                _cfg.clear()
                st.success("Tersimpan: " + "; ".join(ubah))
                st.caption("Scheduler membaca ulang config tiap siklus, "
                           "jadi berlaku pada siklus berikutnya tanpa restart.")
            else:
                st.info("Tidak ada perubahan.")


def tarik_data_sekarang(timeout: int = 900):
    """Jalankan run_once.py sebagai proses terpisah (-> (berhasil, keluaran)).

    Proses terpisah supaya model & memori GPU tidak menumpuk di dashboard.
    """
    import subprocess
    proyek = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        p = subprocess.run([sys.executable, "run_once.py"], cwd=proyek, timeout=timeout,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"Melebihi {timeout // 60} menit — dihentikan."
    keluaran = (p.stdout or "") + (p.stderr or "")
    bersih = [b for b in keluaran.splitlines() if b.strip() and "it/s]" not in b]
    return p.returncode == 0, "\n".join(bersih[-15:])


def panel_kata_kunci():
    """Sunting kata kunci per sumber, tersimpan ke config.yaml (komentar aman)."""
    from core.config_edit import baca_list, set_list

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")
    with st.expander("🔎 Kata kunci", expanded=False):
        with st.form("form_kata_kunci"):
            isian = {}
            for judul, path, bantuan in SUMBER_KATA_KUNCI:
                nilai_cfg = baca_list(cfg_path, path)
                isian[tuple(path)] = st.text_area(
                    judul, "\n".join(nilai_cfg), height=90, help=bantuan,
                    key=f"kk_{'_'.join(path)}_{hash(tuple(nilai_cfg))}")
            simpan = st.form_submit_button("Simpan", width="stretch")
        if simpan:
            ubah = []
            for judul, path, _ in SUMBER_KATA_KUNCI:
                baru = [b.strip().lstrip("#") for b in isian[tuple(path)].splitlines()]
                baru = [b for b in baru if b]
                if baru != baca_list(cfg_path, path):
                    try:
                        set_list(cfg_path, path, baru)
                        ubah.append(f"{judul}: {len(baru)} kata kunci")
                    except Exception as e:
                        st.error(f"{judul} gagal disimpan: {e}")
            if ubah:
                _cfg.clear()                       # config di-cache 60 detik
                st.success("Tersimpan — " + "; ".join(ubah))
                st.caption("Berlaku pada penarikan data berikutnya "
                           "(scheduler membaca ulang config tiap siklus).")
            else:
                st.info("Tidak ada perubahan.")

        st.caption("Kata kunci lama tetap tersimpan di database; mengubahnya "
                   "hanya memengaruhi data yang ditarik berikutnya.")
        if st.button("⬇️ Tarik data sekarang", width="stretch"):
            with st.spinner("Menarik & menganalisis data..."):
                ok, keluaran = tarik_data_sekarang()
            st.code(keluaran or "(tanpa keluaran)")
            if ok:
                load_docs.clear()
                st.success("Selesai. Muat ulang halaman untuk melihat data baru.")
            else:
                st.error("Gagal — lihat pesan di atas.")


# ── Header & filter ─────────────────────────────────────────────
st.title("📡 Monitoring Sosial & Berita")
st.caption("Sentimen · Social Network Analysis · Deteksi Buzzer — gaya Drone Emprit")

if not os.path.isfile(_db_path()):
    st.warning("Belum ada database. Jalankan `python run_once.py` dulu.")
    st.stop()

df = load_docs()
if df.empty:
    st.warning("Database masih kosong. Jalankan `python run_once.py`.")
    st.stop()

def kata_kunci_dokumen(d: pd.DataFrame) -> pd.Series:
    """Daftar kata kunci yang mencocokkan tiap dokumen (dari kolom keywords_matched)."""
    def urai(v):
        try:
            hasil = json.loads(v) if isinstance(v, str) and v.strip() else []
        except Exception:
            hasil = []
        return [str(x) for x in hasil] if isinstance(hasil, list) else []
    return d["keywords_matched"].map(urai) if "keywords_matched" in d.columns \
        else pd.Series([[]] * len(d), index=d.index)


with st.sidebar:
    st.header("Filter")

    # Database menyimpan SEMUA yang pernah ditarik, termasuk kata kunci lama.
    # Tanpa filter ini, mengganti kata kunci tidak mengubah tampilan: topik lama
    # yang jumlahnya jauh lebih banyak akan mendominasi dan menyesatkan.
    df["_kata"] = kata_kunci_dokumen(df)
    semua_kata = sorted({k for daftar in df["_kata"] for k in daftar})
    # Dokumen tanpa kata kunci dibuat terlihat sebagai pilihan tersendiri.
    # Sebelumnya ia selalu ikut tampil diam-diam — berita nyasar jadi terhitung.
    TANPA = "(tanpa kata kunci)"
    if (df["_kata"].map(len) == 0).any():
        semua_kata.append(TANPA)
    kata_aktif = [k for k in (_cfg().get("keywords") or []) if k in semua_kata]
    if semua_kata:
        pakai_aktif = st.checkbox(
            "Hanya kata kunci aktif", value=bool(kata_aktif),
            help="Database memuat semua penarikan sebelumnya. Centang agar hanya "
                 "topik di config.yaml yang ditampilkan.")
        sel_kata = st.multiselect(
            "Kata kunci", semua_kata,
            default=(kata_aktif if (pakai_aktif and kata_aktif) else semua_kata))
        tertinggal = [k for k in semua_kata if k not in sel_kata]
        if tertinggal:
            st.caption(f"{len(tertinggal)} kata kunci lain disembunyikan: "
                       f"{', '.join(tertinggal[:5])}{'...' if len(tertinggal) > 5 else ''}")
    else:
        sel_kata = []

    if df["dt"].notna().any():
        tmin = df["dt"].min().date()
        tmaks = df["dt"].max().date()
        rentang = st.date_input("Rentang tanggal", (tmin, tmaks),
                                min_value=tmin, max_value=tmaks,
                                help="Batasi periode; berguna setelah menarik arsip "
                                     "rentang lain.")
    else:
        rentang = ()

    # Berita sindikasi: satu rilis dimuat banyak media. Bila salinan ikut
    # dihitung, volume & sentimen satu isu terhitung berulang kali.
    n_salinan = int(df["duplikat_dari"].fillna("").astype(str).str.len().gt(0).sum()) \
        if "duplikat_dari" in df.columns else 0
    gabung_sindikasi = st.checkbox(
        f"Gabungkan berita sindikasi ({n_salinan} salinan)", value=bool(n_salinan),
        disabled=not n_salinan,
        help="Satu rilis yang dimuat ulang banyak media dihitung SEKALI. "
             "Tandai dulu lewat: python tools/dedup_sindikasi.py --yakin")

    hitung_per_dokumen = st.checkbox(
        "Hitung kata per dokumen", value=True,
        help="Tiap kata dihitung sekali per berita, supaya satu artikel panjang "
             "tidak mendominasi word cloud.")

    plats = sorted(df["platform"].dropna().unique().tolist())
    sel_plat = st.multiselect("Platform", plats, default=plats)
    sents = ["positive", "neutral", "negative"]
    sel_sent = st.multiselect("Sentimen", sents, default=sents)
    st.divider()
    pakai_stem = st.checkbox("Stemming (Sastrawi)", value=True,
                             help="Kembalikan kata ke bentuk dasar. Matikan bila lambat.")
    st.caption(f"Total dokumen di DB: {len(df)}")
    st.divider()
    panel_kata_kunci()
    panel_arsip()
    panel_rem()

f = df[df["platform"].isin(sel_plat) & df["sentiment_label"].isin(sel_sent)]
if sel_kata:
    pilih = set(sel_kata)
    tanpa_ikut = "(tanpa kata kunci)" in pilih
    f = f[f["_kata"].map(lambda ks: bool(pilih & set(ks)) or (tanpa_ikut and not ks))]
if isinstance(rentang, (tuple, list)) and len(rentang) == 2 and f["dt"].notna().any():
    awal = pd.Timestamp(rentang[0], tz="UTC")
    akhir = pd.Timestamp(rentang[1], tz="UTC") + pd.Timedelta(days=1)
    f = f[f["dt"].isna() | ((f["dt"] >= awal) & (f["dt"] < akhir))]
if gabung_sindikasi and "duplikat_dari" in f.columns:
    f = f[f["duplikat_dari"].fillna("").astype(str).str.len() == 0]
teks_terpilih = tuple(f["teks"].dropna().tolist())

if len(f) != len(df):
    st.caption(f"Menampilkan **{len(f)}** dari {len(df)} dokumen di database "
               f"(filter kata kunci / tanggal / platform / sentimen aktif).")

(tab_ring, tab_sov, tab_topik, tab_teks, tab_heat, tab_sna, tab_bot, tab_ml,
 tab_emo, tab_nuansa, tab_hf, tab_label, tab_entitas) = st.tabs(
    ["📊 Ringkasan", "📣 Share of Voice", "📈 Topik", "☁️ Teks & Word Cloud",
     "🔥 Heatmap", "🕸️ Jaringan", "🤖 Bot/Buzzer", "🧪 Klasifikasi ML",
     "😠 Emosi", "🎯 Intent & Sarkasme", "🔬 Banding Model", "🏷️ Pelabelan",
     "🗣️ Entitas & Kutipan"])


# ── TAB 1: Ringkasan ────────────────────────────────────────────
with tab_ring:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total dokumen", len(f))
    c2.metric("Positif", int((f["sentiment_label"] == "positive").sum()))
    c3.metric("Netral", int((f["sentiment_label"] == "neutral").sum()))
    c4.metric("Negatif", int((f["sentiment_label"] == "negative").sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Volume per hari")
        if f["dt"].notna().any():
            st.line_chart(f.dropna(subset=["dt"]).set_index("dt").resample("D").size())
        else:
            st.info("Tanggal tidak tersedia.")
    with right:
        st.subheader("Distribusi sentimen")
        st.bar_chart(f["sentiment_label"].value_counts())

    st.subheader("Sumber / aktor teratas")
    st.dataframe(f["source"].value_counts().head(15)
                 .rename_axis("source").reset_index(name="jumlah"),
                 width="stretch")

    # Berita panjang dinilai per paragraf — tampilkan bagian yang memicu label negatif.
    if "bagian_total" in f.columns and f["bagian_total"].fillna(0).gt(1).any():
        st.subheader("Bagian paling negatif (berita panjang)")
        st.caption("Berita panjang dinilai per paragraf, bukan hanya alinea pembuka. "
                   "Kolom 'bagian negatif' menunjukkan berapa paragraf bernada negatif "
                   "— berguna untuk membedakan berita yang negatif seluruhnya dari "
                   "berita netral yang memuat satu kutipan keras.")
        panjang = f[f["bagian_total"].fillna(0) > 1].copy()
        panjang["bagian negatif"] = (panjang["bagian_negatif"].fillna(0).astype(int).astype(str)
                                     + " / " + panjang["bagian_total"].astype(int).astype(str))
        # judul & kutipan dipendekkan agar semua kolom muat tanpa geser ke samping
        panjang["judul"] = panjang["title"].fillna("").str.slice(0, 60)
        panjang["kutipan paling negatif"] = panjang["kutipan_negatif"].fillna("").str.slice(0, 90)
        st.dataframe(
            panjang.sort_values("bagian_negatif", ascending=False)
            .head(25)[["source", "judul", "sentiment_label", "sentiment_score",
                       "bagian negatif", "kutipan paling negatif"]]
            .rename(columns={"sentiment_label": "sentimen", "sentiment_score": "nilai"}),
            width="stretch", hide_index=True)


# ── TAB: Share of Voice & laporan ───────────────────────────────
with tab_sov:
    st.subheader("📣 Share of Voice")
    st.caption("Porsi pemberitaan tiap pihak/isu beserta nadanya. Dihitung per "
               "DOKUMEN (bukan per kemunculan kata) dan mengikuti filter di "
               "sidebar, termasuk penggabungan berita sindikasi.")

    from analysis import sov as _sov

    entitas_teks = st.text_area(
        "Pihak / isu yang dibandingkan (satu per baris)",
        "\n".join(_cfg().get("keywords") or []), height=90, key="sov_entitas",
        help="Boleh apa saja, tidak harus kata kunci scraping — mis. nama tokoh, "
             "lembaga, atau merek yang ingin dibandingkan porsinya.")
    entitas = [b.strip() for b in entitas_teks.splitlines() if b.strip()]

    saring = st.checkbox(
        "Saring berita yang hanya menyinggung sekilas", value=True,
        help="Berita dianggap relevan bila kata kunci ada di judul atau disebut "
             "minimal 2 kali. Tanpa ini, 'dolar AS' di berita judi ikut terhitung.")

    if not entitas:
        st.info("Isi minimal satu pihak/isu di atas.")
    elif f.empty:
        st.info("Tidak ada dokumen pada filter saat ini.")
    else:
        dasar = f.copy()
        dibuang = 0
        if saring:
            dasar, dibuang = _sov.saring_relevan(dasar, entitas)
        if dasar.empty:
            st.warning("Semua dokumen tersaring. Longgarkan filter atau matikan "
                       "saringan relevansi.")
        else:
            tabel_sov = _sov.hitung(dasar, entitas)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dokumen dianalisis", len(dasar))
            c2.metric("Disaring (sekilas)", dibuang)
            c3.metric("Media", int(dasar["source"].nunique()))
            st.dataframe(tabel_sov, width="stretch", hide_index=True)
            st.caption("Satu berita bisa menyebut beberapa pihak, jadi jumlah "
                       "share bisa melebihi 100%. Indeks nada: "
                       "(positif − negatif) ÷ jumlah berita.")

            from analysis import media as _media
            tier = _media.ringkas_tier(dasar)
            if tier:
                st.markdown("**Mutu sumber (tier media)**")
                st.dataframe(pd.DataFrame(tier), width="stretch", hide_index=True)
                st.caption("Tier 1 = nasional arus utama, 2 = menengah/vertikal, "
                           "3 = lainnya. Bobot ini PERKIRAAN reputasi, bukan data "
                           "traffic — ubah di resources/media_tier.csv.")

            tren_sov = _sov.tren(dasar, entitas)
            if not tren_sov.empty and len(tren_sov) > 1:
                st.markdown("**Tren jumlah berita per hari**")
                st.line_chart(tren_sov)

            lonj = _sov.lonjakan(dasar)
            if lonj and lonj.get("lonjakan"):
                st.warning(f"⚠️ Lonjakan volume {lonj['tanggal']}: {lonj['jumlah']} "
                           f"berita, {lonj['rasio']}× rata-rata ({lonj['rata_rata']}/hari).")
            elif lonj and lonj.get("catatan"):
                st.info(f"Deteksi lonjakan belum berlaku: {lonj['catatan']}.")

    st.divider()
    st.markdown("**📄 Laporan otomatis**")
    st.caption("Membuat folder berisi ringkasan siap kirim, Excel, grafik, dan "
               "halaman HTML yang bisa dicetak jadi PDF.")
    hari_lap = st.number_input("Rentang (hari ke belakang)", 1, 365, 7, key="lap_hari")
    if st.button("Buat laporan", width="stretch", key="lap_buat"):
        import subprocess
        proyek = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with st.spinner("Menyusun laporan..."):
            p = subprocess.run(
                [sys.executable, "-u", "tools/laporan.py", "--hari", str(hari_lap),
                 *(["--entitas", *entitas] if entitas else [])],
                cwd=proyek, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=1800)
        st.code((p.stdout or "") + (p.stderr or ""))
        if p.returncode == 0:
            import datetime as _dt
            folder = os.path.join(proyek, "laporan", _dt.date.today().isoformat())
            for nama in ("ringkasan.md", "laporan.xlsx", "laporan.html"):
                jalur = os.path.join(folder, nama)
                if os.path.isfile(jalur):
                    with open(jalur, "rb") as fh:
                        st.download_button(f"⬇️ {nama}", fh.read(), file_name=nama,
                                           key=f"unduh_{nama}")
        else:
            st.error("Gagal membuat laporan — lihat pesan di atas.")


# ── TAB: Entitas & Kutipan ──────────────────────────────────────
with tab_entitas:
    st.subheader("🗣️ Tokoh, lembaga & kutipan")
    st.caption("Siapa yang muncul di pemberitaan, dan kalimat apa yang dikutip. "
               "Diambil dengan ATURAN (pola kutipan bahasa Indonesia), bukan "
               "model NER — hasilnya bisa diperiksa, tapi kutipan tanpa tanda "
               "kutip atau nama tak lazim bisa terlewat.")
    if f.empty:
        st.info("Tidak ada dokumen pada filter saat ini.")
    else:
        from analysis import entitas as _ent
        batas = st.slider("Jumlah berita yang dipindai", 20, 500,
                          min(200, len(f)), step=20, key="ent_batas",
                          help="Makin banyak makin lambat.")
        with st.spinner("Memindai teks..."):
            kutip, tabel_ent = _ent.dari_dokumen(f.head(batas))

        k1, k2 = st.columns(2)
        with k1:
            st.markdown("**Tokoh & lembaga paling sering disebut**")
            if tabel_ent:
                st.dataframe(pd.DataFrame(tabel_ent, columns=["nama", "sebutan"]),
                             width="stretch", hide_index=True, height=380)
            else:
                st.info("Belum ada entitas terdeteksi.")
        with k2:
            st.markdown("**Pembicara paling sering dikutip**")
            if kutip:
                from collections import Counter as _C
                bicara = _C(k["pembicara"] for k in kutip).most_common(20)
                st.dataframe(pd.DataFrame(bicara, columns=["pembicara", "kutipan"]),
                             width="stretch", hide_index=True, height=380)
            else:
                st.info("Belum ada kutipan terdeteksi.")

        if kutip:
            st.markdown(f"**Kutipan ({len(kutip)} ditemukan)**")
            cari = st.text_input("Saring menurut nama pembicara", key="ent_cari")
            tampil = [k for k in kutip
                      if not cari or cari.lower() in k["pembicara"].lower()]
            if tampil:
                st.dataframe(pd.DataFrame(tampil)[["pembicara", "kutipan", "source",
                                                   "sentimen", "judul"]],
                             width="stretch", hide_index=True, height=420)
            else:
                st.info("Tidak ada pembicara yang cocok.")


# ── TAB 2: Tren per topik ───────────────────────────────────────
def explode_topics(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in d.iterrows():
        try:
            kws = json.loads(r.get("keywords_matched") or "[]")
        except Exception:
            kws = []
        for kw in kws:
            rows.append({"dt": r["dt"], "topic": kw,
                         "sentiment_label": r["sentiment_label"]})
    return pd.DataFrame(rows)


with tab_topik:
    st.subheader("📈 Tren sentimen per topik")
    topics_df = explode_topics(f)
    if topics_df.empty or topics_df["dt"].isna().all():
        st.info("Belum ada data topik/tanggal. Pastikan `keywords` terisi di config.yaml.")
    else:
        topics_df = topics_df.dropna(subset=["dt"])
        all_topics = sorted(topics_df["topic"].unique().tolist())
        c1, c2 = st.columns([3, 1])
        with c1:
            sel_topics = st.multiselect("Topik", all_topics, default=all_topics[:5])
        with c2:
            freq = st.selectbox("Granularitas", ["Harian", "Mingguan"], index=0)
        rule = "D" if freq == "Harian" else "W"

        tf = topics_df[topics_df["topic"].isin(sel_topics)].copy()
        if tf.empty:
            st.info("Pilih minimal satu topik.")
        else:
            tf["val"] = tf["sentiment_label"].map(_SENT_VAL).fillna(0)
            st.markdown("**Indeks sentimen bersih** (1 = sangat positif, −1 = sangat negatif)")
            net = tf.set_index("dt").groupby("topic")["val"].resample(rule).mean().reset_index()
            if not net.empty:
                st.line_chart(net.pivot(index="dt", columns="topic", values="val"))

            st.markdown("**Volume perbincangan per topik**")
            vol = tf.set_index("dt").groupby("topic")["val"].resample(rule).size().reset_index(name="jumlah")
            st.area_chart(vol.pivot(index="dt", columns="topic", values="jumlah").fillna(0))

            st.markdown("**Rincian sentimen per topik**")
            pivot = tf.groupby(["topic", "sentiment_label"]).size().unstack(fill_value=0)
            for c in ("positive", "neutral", "negative"):
                if c not in pivot.columns:
                    pivot[c] = 0
            pivot["total"] = pivot[["positive", "neutral", "negative"]].sum(axis=1)
            pivot["net %"] = ((pivot["positive"] - pivot["negative"]) /
                              pivot["total"].replace(0, 1) * 100).round(1)
            st.dataframe(pivot[["positive", "neutral", "negative", "total", "net %"]]
                         .sort_values("total", ascending=False), width="stretch")


# ── TAB 3: Teks & Word Cloud ────────────────────────────────────
with tab_teks:
    st.subheader("☁️ Word cloud & frekuensi kata")
    st.caption("Teks melewati preprocessing: hapus URL/mention/emoji, normalisasi kata baku, "
               "stopword removal, stemming.")

    if not teks_terpilih:
        st.info("Tidak ada teks pada filter ini.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            pilih_sent = st.selectbox("Word cloud untuk sentimen",
                                      ["Semua", "positive", "neutral", "negative"])
        with c2:
            top_n = st.slider("Jumlah kata teratas", 10, 100, 40, step=5)

        subset = f if pilih_sent == "Semua" else f[f["sentiment_label"] == pilih_sent]
        texts_sub = tuple(subset["teks"].dropna().tolist())

        if not texts_sub:
            st.info("Tidak ada data untuk pilihan ini.")
        else:
            with st.spinner("Memproses teks..."):
                freq = cached_freq(texts_sub, pakai_stem, hitung_per_dokumen)

            if not freq:
                st.info("Tidak ada kata tersisa setelah preprocessing.")
            else:
                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt
                    wc = WordCloud(width=1200, height=500, background_color="white",
                                   colormap="viridis", max_words=200
                                   ).generate_from_frequencies(freq)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Word cloud tidak tersedia ({e}). Menampilkan tabel saja.")

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Kata paling sering**")
                    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    st.bar_chart(pd.DataFrame(top, columns=["kata", "jumlah"])
                                 .set_index("kata"))
                with cc2:
                    st.markdown("**Frasa (bigram) paling sering**")
                    bg = textstats.ngrams(list(texts_sub), 2, get_preprocessor(pakai_stem), 20)
                    if bg:
                        st.bar_chart(pd.DataFrame(bg, columns=["frasa", "jumlah"])
                                     .set_index("frasa"))
                    else:
                        st.caption("Belum cukup data untuk bigram.")

        st.divider()
        st.markdown("**Kata khas negatif vs positif** — kata yang membedakan kedua kelompok")
        neg_t = f[f["sentiment_label"] == "negative"]["teks"].dropna().tolist()
        pos_t = f[f["sentiment_label"] == "positive"]["teks"].dropna().tolist()
        if neg_t and pos_t:
            pp = get_preprocessor(pakai_stem)
            d1, d2 = st.columns(2)
            with d1:
                st.caption("Khas NEGATIF")
                st.dataframe(pd.DataFrame(
                    textstats.distinctive_terms(neg_t, pos_t, pp, 15),
                    columns=["kata", "kekhasan", "jumlah"]), width="stretch")
            with d2:
                st.caption("Khas POSITIF")
                st.dataframe(pd.DataFrame(
                    textstats.distinctive_terms(pos_t, neg_t, pp, 15),
                    columns=["kata", "kekhasan", "jumlah"]), width="stretch")
        else:
            st.caption("Butuh dokumen positif dan negatif untuk perbandingan ini.")


# ── TAB 4: Heatmap ──────────────────────────────────────────────
def draw_heatmap(matrix: pd.DataFrame, judul: str, fmt: str = "{:.0f}"):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(min(1 + matrix.shape[1] * 0.7, 14),
                                        min(1 + matrix.shape[0] * 0.5, 10)))
        im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(matrix.shape[0]), matrix.index)
        ax.set_title(judul)
        fig.colorbar(im, ax=ax, shrink=0.8)
        # anotasi angka bila matriks tidak terlalu besar
        if matrix.size <= 200:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v = matrix.values[i, j]
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except Exception as e:
        st.warning(f"Heatmap gagal digambar ({e}) — menampilkan tabel.")
        st.dataframe(matrix, width="stretch")


with tab_heat:
    st.subheader("🔥 Peta panas aktivitas")
    fh = f.dropna(subset=["dt"]).copy()
    if fh.empty:
        st.info("Tidak ada data bertanggal.")
    else:
        fh["jam"] = fh["dt"].dt.hour
        fh["hari_idx"] = fh["dt"].dt.dayofweek
        fh["hari"] = fh["hari_idx"].map(dict(enumerate(HARI)))

        st.markdown("**Jam × Hari** — kapan perbincangan memuncak "
                    "(aktivitas merata 24 jam = indikasi bot)")
        m = (fh.pivot_table(index="hari", columns="jam", values="doc_id",
                            aggfunc="count", fill_value=0)
               .reindex(HARI).fillna(0))
        m = m.reindex(columns=range(24), fill_value=0)
        draw_heatmap(m, "Jumlah dokumen per jam dan hari")

        st.divider()
        st.markdown("**Topik × Sentimen**")
        td = explode_topics(fh)
        if td.empty:
            st.caption("Belum ada topik terdeteksi.")
        else:
            m2 = td.pivot_table(index="topic", columns="sentiment_label",
                                values="dt", aggfunc="count", fill_value=0)
            for c in ("negative", "neutral", "positive"):
                if c not in m2.columns:
                    m2[c] = 0
            draw_heatmap(m2[["negative", "neutral", "positive"]],
                         "Jumlah dokumen per topik dan sentimen")


# ── TAB 5: SNA ──────────────────────────────────────────────────
with tab_sna:
    st.subheader("🕸️ Social Network Analysis")
    plat_sna = "twitter" if "twitter" in sel_plat else (sel_plat[0] if sel_plat else None)
    G = sna.build_graph(_db_path(), platform=plat_sna)

    if G.number_of_nodes() == 0:
        st.info("Belum ada data interaksi. Jalankan collector X/Instagram untuk mengisi graf.")
    else:
        s = sna.summary(G)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Aktor (node)", s["nodes"])
        m2.metric("Interaksi (edge)", s["edges"])
        m3.metric("Komunitas", s["communities"])
        m4.metric("Densitas", s["density"])

        parts = sna.detect_communities(G)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Paling berpengaruh (degree)**")
            st.dataframe(pd.DataFrame(sna.top_actors(G, "degree", 15),
                                      columns=["aktor", "skor"]), width="stretch")
        with cb:
            st.markdown("**Jembatan antar-klaster (betweenness)**")
            bt = [(a, round(v, 4)) for a, v in sna.top_actors(G, "betweenness", 15)]
            st.dataframe(pd.DataFrame(bt, columns=["aktor", "skor"]), width="stretch")

        st.markdown("**Peta jaringan (top 40 aktor)**")
        top_nodes = [a for a, _ in sna.top_actors(G, "degree", 40)]
        sub = G.subgraph(top_nodes)
        palette = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                   "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
        dot = ["digraph { rankdir=LR; node [style=filled, fontsize=10];"]
        for n in sub.nodes():
            color = palette[parts.get(n, 0) % len(palette)]
            dot.append(f'"{n.replace(chr(34), "")}" [fillcolor="{color}"];')
        for u, v in sub.edges():
            dot.append(f'"{u.replace(chr(34), "")}" -> "{v.replace(chr(34), "")}";')
        dot.append("}")
        st.graphviz_chart("\n".join(dot))


# ── TAB 6: Bot / Buzzer ─────────────────────────────────────────
with tab_bot:
    st.subheader("🤖 Deteksi bot & buzzer")
    st.caption("Skor heuristik dari perilaku: frekuensi posting, konten duplikat, "
               "sebaran jam aktif, pola username, rasio non-orisinal.")

    plat_bot = st.selectbox("Platform", plats,
                            index=plats.index("twitter") if "twitter" in plats else 0)
    if st.button("Jalankan analisis bot", type="primary"):
        from analysis import bot_detect
        with st.spinner("Menganalisis perilaku akun..."):
            aktor = bot_detect.analyze_actors(_db_path(), platform=plat_bot)
            klaster = bot_detect.coordinated_clusters(_db_path(), platform=plat_bot)

        if not aktor:
            st.info("Belum cukup data akun (minimal 3 posting per akun).")
        else:
            ad = pd.DataFrame(aktor)
            ad["flags"] = ad["flags"].apply(lambda x: "; ".join(x) if x else "-")
            k1, k2, k3 = st.columns(3)
            k1.metric("Akun dianalisis", len(ad))
            k2.metric("Indikasi tinggi", int((ad["kategori"] == "tinggi").sum()))
            k3.metric("Indikasi sedang", int((ad["kategori"] == "sedang").sum()))
            st.dataframe(ad[["actor", "n_posts", "posts_per_day", "dup_ratio",
                             "hour_coverage", "non_original", "bot_score",
                             "kategori", "flags"]], width="stretch")

        st.divider()
        st.markdown("**Posting serentak (coordinated behavior)** — teks identik "
                    "disebar banyak akun dalam waktu berdekatan")
        if klaster:
            kd = pd.DataFrame(klaster)
            kd["actors"] = kd["actors"].apply(lambda x: ", ".join(x))
            st.dataframe(kd[["n_actors", "n_posts", "rentang_menit",
                             "contoh_teks", "actors"]], width="stretch")
        else:
            st.caption("Tidak ditemukan pola posting serentak.")
    else:
        st.info("Klik tombol di atas untuk menjalankan analisis.")


# ── TAB 7: Klasifikasi ML ───────────────────────────────────────
with tab_ml:
    st.subheader("🧪 Klasifikasi sentimen: TF-IDF + SMOTE")
    st.caption("Membandingkan 6 algoritma sebelum dan sesudah penyeimbangan data (SMOTE). "
               "Label latih diambil dari kolom sentimen di database.")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Proporsi data uji", 0.1, 0.4, 0.2, step=0.05)
    with c2:
        max_feat = st.select_slider("Maks fitur TF-IDF", [1000, 2000, 5000, 10000], value=5000)
    with c3:
        ngram_max = st.selectbox("N-gram maksimum", [1, 2, 3], index=1)

    if st.button("Latih & bandingkan model", type="primary"):
        from analysis.ml_classify import load_dataset_from_db, run_comparison, summary_table
        plat_ml = None if len(sel_plat) == len(plats) else (sel_plat[0] if sel_plat else None)
        texts, labels = load_dataset_from_db(_db_path(), platform=plat_ml)

        if not texts:
            st.warning("Belum ada data berlabel sentimen. Jalankan `python run_once.py` dulu.")
        else:
            with st.spinner(f"Melatih 6 model pada {len(texts)} dokumen..."):
                res = run_comparison(texts, labels, test_size=test_size,
                                     max_features=max_feat, ngram_max=ngram_max,
                                     p=get_preprocessor(pakai_stem))
            if "error" in res:
                st.error(res["error"])
            else:
                a, b, c = st.columns(3)
                a.metric("Dokumen", res["n_dokumen"])
                b.metric("Latih / Uji", f"{res['n_latih']} / {res['n_uji']}")
                c.metric("Fitur TF-IDF", res["fitur_tfidf"])
                st.caption(f"SMOTE: {res['smote']}")

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Distribusi sebelum SMOTE**")
                    st.bar_chart(pd.Series(res["distribusi_sebelum_smote"]))
                with d2:
                    st.markdown("**Distribusi sesudah SMOTE**")
                    st.bar_chart(pd.Series(res["distribusi_sesudah_smote"]))

                st.markdown("**Perbandingan model (urut F1 sesudah SMOTE)**")
                st.dataframe(pd.DataFrame(summary_table(res)), width="stretch")

                st.markdown("**Confusion matrix (sesudah SMOTE)**")
                nama_model = st.selectbox("Model", list(res["hasil"].keys()))
                cm = res["hasil"][nama_model].get("after", {})
                if "confusion_matrix" in cm:
                    mat = pd.DataFrame(cm["confusion_matrix"],
                                       index=[f"asli: {l}" for l in cm["labels"]],
                                       columns=[f"prediksi: {l}" for l in cm["labels"]])
                    draw_heatmap(mat, f"Confusion matrix — {nama_model}")
    else:
        st.info("Atur parameter lalu klik tombol untuk melatih model.")

# ── TAB 8: Emosi ────────────────────────────────────────────────
with tab_emo:
    st.subheader("😠 Analisis emosi")
    st.caption("Lebih rinci dari positif/negatif: marah, takut, sedih, senang, cinta. "
               "Dua berita sama-sama negatif bisa berbeda — yang memicu KEMARAHAN "
               "biasanya lebih cepat viral daripada yang memicu kesedihan.")

    punya_emosi = ("emotion_label" in f.columns and
                   f["emotion_label"].notna().any() and
                   (f["emotion_label"].astype(str).str.len() > 0).any())

    if st.button("Analisis emosi dokumen", type="primary"):
        from analysis.emotion import analisis_db
        with st.spinner("Menganalisis emosi (model diunduh sekali bila belum ada)..."):
            n = analisis_db(_db_path(), limit=2000)
        # if/else biasa — BUKAN ekspresi kondisional sebagai pernyataan. Streamlit
        # "magic" akan menulis nilai ekspresi telanjang, lalu mencoba membaca nama
        # variabel dari SATU baris sumber; pernyataan multi-baris membuatnya gagal
        # parse (SyntaxError "'(' was never closed").
        if n:
            st.success(f"{n} dokumen dianalisis.")
        else:
            st.warning("Tidak ada yang dianalisis. Pastikan model emosi tersedia: "
                       "`python tools/setup_local_models.py --emotion`")
        load_docs.clear()
        st.rerun()

    if not punya_emosi:
        st.info("Belum ada data emosi. Klik tombol di atas untuk menganalisis.")
    else:
        fe = f[f["emotion_label"].astype(str).str.len() > 0].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribusi emosi**")
            st.bar_chart(fe["emotion_label"].value_counts())
        with c2:
            st.markdown("**Emosi per hari**")
            fed = fe.dropna(subset=["dt"])
            if not fed.empty:
                tren = (fed.set_index("dt").groupby("emotion_label")
                          .resample("D").size().reset_index(name="jumlah"))
                st.line_chart(tren.pivot(index="dt", columns="emotion_label",
                                         values="jumlah").fillna(0))

        st.markdown("**Emosi × Sentimen**")
        m = fe.pivot_table(index="emotion_label", columns="sentiment_label",
                           values="doc_id", aggfunc="count", fill_value=0)
        draw_heatmap(m, "Jumlah dokumen per emosi dan sentimen")

        st.markdown("**Akun/sumber dengan emosi marah terbanyak**")
        marah = fe[fe["emotion_label"] == "marah"]
        if not marah.empty:
            st.dataframe(marah["source"].value_counts().head(10)
                         .rename_axis("sumber").reset_index(name="jumlah"),
                         width="stretch")
        else:
            st.caption("Tidak ada dokumen berlabel 'marah' pada filter ini.")


# ── TAB: Intent · Intensitas · Sarkasme ─────────────────────────
def _terisi(kolom):
    return (kolom in f.columns and f[kolom].notna().any() and
            (f[kolom].astype(str).str.len() > 0).any())


with tab_nuansa:
    st.subheader("🎯 Intent, intensitas & sarkasme")
    st.caption("Lapisan di atas sentimen: APA tujuan orang menulis (intent), SEBERAPA "
               "kuat sikapnya (5 tingkat), dan apakah ia SARKAS — pujian yang "
               "sebenarnya sindiran bisa membalik makna sentimen.")

    b1, b2, b3 = st.columns(3)
    jalankan = None
    if b1.button("Analisis intent (zero-shot)"):
        jalankan = ("intent",)
    if b2.button("Analisis intensitas 5 tingkat"):
        jalankan = ("intensitas",)
    if b3.button("Deteksi sarkasme"):
        jalankan = ("sarkasme",)

    if jalankan:
        with st.spinner("Menganalisis... (model diunduh sekali bila belum ada)"):
            try:
                if jalankan == ("sarkasme",):
                    from analysis.sarcasm import analisis_db as sarkas_db
                    n = sarkas_db(_db_path(), limit=2000)
                else:
                    from analysis.zeroshot import analisis_db as zs_db
                    n = zs_db(_db_path(), tugas=jalankan, limit=500).get(jalankan[0], 0)
                st.success(f"{n} dokumen dianalisis.")
            except Exception as e:
                st.error(f"Gagal: {e}")
        load_docs.clear()
        st.rerun()

    st.info("Akurasi terukur: **sarkasme** F1 0,727 (538 tweet uji resmi) · "
            "**intensitas** tepat 65,5% / meleset ≤1 tingkat 88,7% (PRDECT-ID). "
            "**Intent belum bisa diukur** — belum ada data berlabel yang relevan; "
            "anggap sebagai indikasi, bukan kepastian.")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Intent**")
        if _terisi("intent_label"):
            st.bar_chart(f["intent_label"].replace("", pd.NA).dropna().value_counts())
        else:
            st.caption("Belum dianalisis.")
    with k2:
        st.markdown("**Intensitas (5 tingkat)**")
        if _terisi("intensitas_label"):
            urutan = ["sangat negatif", "negatif", "netral", "positif", "sangat positif"]
            st.bar_chart(f["intensitas_label"].value_counts().reindex(urutan).fillna(0))
        else:
            st.caption("Belum dianalisis. Pastikan ambang sudah dikalibrasi: "
                       "`python -m analysis.zeroshot --kalibrasi`")

    if _terisi("intent_label"):
        st.markdown("**Intent × Sentimen**")
        fi = f[f["intent_label"].astype(str).str.len() > 0]
        draw_heatmap(fi.pivot_table(index="intent_label", columns="sentiment_label",
                                    values="doc_id", aggfunc="count", fill_value=0),
                     "Jumlah dokumen per intent dan sentimen")

    if _terisi("sarkasme_label"):
        fs = f[f["sarkasme_label"].astype(str).str.len() > 0]
        sarkas = fs[fs["sarkasme_label"] == "sarkas"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Dokumen dicek", len(fs))
        m2.metric("Terdeteksi sarkas", len(sarkas))
        m3.metric("Proporsi", f"{len(sarkas) / max(len(fs), 1):.0%}")
        st.markdown("**Sarkas tapi dilabeli POSITIF** — kandidat salah baca sentimen")
        curiga = sarkas[sarkas["sentiment_label"] == "positive"]
        if not curiga.empty:
            st.dataframe(curiga[["source", "teks", "sarkasme_score"]]
                         .sort_values("sarkasme_score", ascending=False).head(20),
                         width="stretch")
        else:
            st.caption("Tidak ada.")


# ── TAB 9: Banding model HuggingFace ────────────────────────────
with tab_hf:
    st.subheader("🔬 Bandingkan model sentimen")
    st.caption("Menguji beberapa model Indonesia dari HuggingFace pada data "
               "BERLABEL MANUSIA, supaya perbandingannya sahih.")

    from analysis.hf_models import list_model, compare_models, tabel_ringkas
    from analysis.hf_datasets import list_sumber, load_labeled

    daftar = list_model()
    st.markdown("**Model yang tersedia**")
    st.dataframe(pd.DataFrame(daftar), width="stretch")

    c1, c2, c3 = st.columns(3)
    with c1:
        pilih_model = st.multiselect("Model diuji", [m["kunci"] for m in daftar],
                                     default=[m["kunci"] for m in daftar][:2])
    with c2:
        sumber = st.selectbox("Dataset uji (berlabel manusia)",
                              [d["kunci"] for d in list_sumber()])
    with c3:
        n_uji = st.slider("Jumlah contoh uji", 50, 500, 150, step=50)

    st.caption("Catatan: model diunduh saat pertama dipakai (±500 MB per model).")

    if st.button("Jalankan perbandingan", type="primary"):
        if not pilih_model:
            st.warning("Pilih minimal satu model.")
        else:
            with st.spinner("Mengambil data uji berlabel..."):
                teks_u, label_u = load_labeled(sumber, limit=n_uji * 4, seimbang=True)
            if not teks_u:
                st.error("Dataset uji gagal dimuat (cek koneksi internet).")
            else:
                with st.spinner(f"Menguji {len(pilih_model)} model..."):
                    res = compare_models(teks_u, label_u, pilih_model, max_sample=n_uji)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.markdown(f"**Hasil** — {res['n_uji']} contoh uji, "
                                f"kelas: {', '.join(res['kelas'])}")
                    st.dataframe(pd.DataFrame(tabel_ringkas(res)), width="stretch")
                    if res.get("terbaik"):
                        st.success(f"Terbaik: **{res['terbaik']}** — "
                                   f"atur di config.yaml → sentiment.model_dir "
                                   f"atau unduh model tersebut.")
                    gagal = [k for k, v in res["hasil"].items() if "error" in v]
                    if gagal:
                        st.warning(f"Gagal dimuat (biasanya koneksi): {', '.join(gagal)}")


# ── TAB: Pelabelan manual ───────────────────────────────────────
FILE_LABEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "label_berita.csv")
LABEL_PILIHAN = [("negative", "🔴 Negatif"), ("neutral", "⚪ Netral"),
                 ("positive", "🟢 Positif")]


def _baca_label() -> pd.DataFrame:
    return pd.read_csv(FILE_LABEL, dtype=str, keep_default_na=False)


def _simpan_label(df: pd.DataFrame) -> None:
    """Tulis seluruh berkas tiap kali — 300 baris, murah, dan tahan crash."""
    df.to_csv(FILE_LABEL, index=False, encoding="utf-8")


with tab_label:
    st.subheader("🏷️ Pelabelan berita oleh manusia")
    st.caption("Label dari MANUSIA adalah satu-satunya cara memperbaiki model di "
               "bahasa berita — melatih dengan tebakan mesin hanya menyalin "
               "kesalahannya. Usulan di bawah hanya tebakan model; kamu yang memutuskan.")

    if not os.path.isfile(FILE_LABEL):
        st.info("Belum ada daftar paragraf. Buat dulu di terminal:\n\n"
                "`python tools/siapkan_pelabelan.py`")
    else:
        df = _baca_label()
        terisi = (df["label"].str.len() > 0)
        n_total, n_isi = len(df), int(terisi.sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Sudah dilabeli", f"{n_isi} / {n_total}")
        c2.metric("Bagian uji", f"{int((terisi & (df['bagian'] == 'uji')).sum())} / "
                                f"{int((df['bagian'] == 'uji').sum())}")
        c3.metric("Bagian latih", f"{int((terisi & (df['bagian'] == 'latih')).sum())} / "
                                  f"{int((df['bagian'] == 'latih').sum())}")
        st.progress(n_isi / n_total if n_total else 0.0)

        belum = df[~terisi]
        if belum.empty:
            st.success("Semua paragraf sudah dilabeli. Lanjut fine-tune:")
            st.code("python tools/siapkan_pelabelan.py --ekspor\n"
                    "python -m analysis.finetune_indobert --tugas sentimen "
                    "--csv data/latih_berita.csv --epochs 4 "
                    "--output models/sentimen-berita-kandidat\n"
                    "python -m analysis.hf_models --bandingkan models/sentimen-berita-kandidat")
        else:
            baris = belum.iloc[0]
            st.caption(f"#{baris['id']} · {baris['bagian']} · {baris['source']}")
            st.markdown(f"> {baris['paragraf']}")
            if baris["url"]:
                st.caption(f"[sumber berita]({baris['url']})")

            kolom = st.columns(len(LABEL_PILIHAN) + 2)
            for kol, (nilai, teks) in zip(kolom, LABEL_PILIHAN):
                utama = "primary" if nilai == baris["usulan"] else "secondary"
                if kol.button(teks, key=f"lab_{nilai}", type=utama, width="stretch"):
                    df.loc[df["id"] == baris["id"], "label"] = nilai
                    _simpan_label(df)
                    st.rerun()
            if kolom[-2].button("⏭️ Lewati", key="lab_skip", width="stretch",
                                help="Paragraf tidak jelas / bukan opini — tandai agar dibuang"):
                df.loc[df["id"] == baris["id"], "label"] = "lewati"
                _simpan_label(df)
                st.rerun()
            if kolom[-1].button("↩️ Batalkan", key="lab_undo", width="stretch",
                                help="Hapus label terakhir yang tersimpan"):
                sudah = df[df["label"].str.len() > 0]
                if not sudah.empty:
                    df.loc[df.index == sudah.index[-1], "label"] = ""
                    _simpan_label(df)
                st.rerun()
            st.caption(f"Usulan model: **{baris['usulan'] or '-'}** (tombolnya disorot). "
                       "Tekan tombol lain bila menurutmu keliru — justru koreksi itu "
                       "yang paling berharga untuk model.")

        with st.expander("Pedoman singkat"):
            st.markdown(
                "- **Negatif** — menyatakan masalah, kerugian, kritik, protes, "
                "kekhawatiran, atau kegagalan.\n"
                "- **Positif** — dukungan, pujian, keberhasilan, harapan, perbaikan.\n"
                "- **Netral** — fakta, prosedur, jadwal, angka, kutipan tanpa sikap.\n"
                "- Nilai **nada paragrafnya**, bukan pendapatmu soal topiknya.\n"
                "- Ragu antara netral dan berpolaritas? Pilih **netral**.\n"
                "- Bukan kalimat berita (navigasi, potongan rusak)? **Lewati**.")


st.caption("Data auto-refresh tiap 60 detik. Atur sumber & kata kunci di config.yaml.")
