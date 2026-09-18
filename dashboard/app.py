"""Dashboard monitoring (Streamlit).

Tab:
  Ringkasan · Topik · Teks & Word Cloud · Heatmap · Jaringan (SNA) ·
  Bot/Buzzer · Klasifikasi ML

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
def cached_freq(texts: tuple, stem: bool) -> dict:
    return textstats.freq_dict(list(texts), get_preprocessor(stem))


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

with st.sidebar:
    st.header("Filter")
    plats = sorted(df["platform"].dropna().unique().tolist())
    sel_plat = st.multiselect("Platform", plats, default=plats)
    sents = ["positive", "neutral", "negative"]
    sel_sent = st.multiselect("Sentimen", sents, default=sents)
    st.divider()
    pakai_stem = st.checkbox("Stemming (Sastrawi)", value=True,
                             help="Kembalikan kata ke bentuk dasar. Matikan bila lambat.")
    st.caption(f"Total dokumen di DB: {len(df)}")

f = df[df["platform"].isin(sel_plat) & df["sentiment_label"].isin(sel_sent)]
teks_terpilih = tuple(f["teks"].dropna().tolist())

tab_ring, tab_topik, tab_teks, tab_heat, tab_sna, tab_bot, tab_ml = st.tabs(
    ["📊 Ringkasan", "📈 Topik", "☁️ Teks & Word Cloud", "🔥 Heatmap",
     "🕸️ Jaringan", "🤖 Bot/Buzzer", "🧪 Klasifikasi ML"])


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
                freq = cached_freq(texts_sub, pakai_stem)

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

st.caption("Data auto-refresh tiap 60 detik. Atur sumber & kata kunci di config.yaml.")
