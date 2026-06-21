"""Dashboard monitoring (Streamlit) — Fase 3.

Jalankan:
    pip install streamlit
    streamlit run dashboard/app.py

Membaca data/monitoring.db (read-only) dan menyajikan: volume, sentimen,
sumber teratas, serta SNA (top aktor, komunitas, graf jaringan).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import pandas as pd
import streamlit as st
import yaml

# agar bisa import paket analysis/ saat dijalankan via streamlit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import sna  # noqa: E402

st.set_page_config(page_title="Monitoring Sosial", layout="wide")


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
    return df


st.title("📡 Monitoring Sosial & Berita")
st.caption("Sentimen + Social Network Analysis — gaya Drone Emprit")

if not os.path.isfile(_db_path()):
    st.warning("Belum ada database. Jalankan `python run_once.py` dulu.")
    st.stop()

df = load_docs()
if df.empty:
    st.warning("Database masih kosong.")
    st.stop()

# ── Filter ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter")
    plats = sorted(df["platform"].dropna().unique().tolist())
    sel_plat = st.multiselect("Platform", plats, default=plats)
    sents = ["positive", "neutral", "negative"]
    sel_sent = st.multiselect("Sentimen", sents, default=sents)

f = df[df["platform"].isin(sel_plat) & df["sentiment_label"].isin(sel_sent)]

# ── Ringkasan ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total dokumen", len(f))
c2.metric("Positif", int((f["sentiment_label"] == "positive").sum()))
c3.metric("Netral", int((f["sentiment_label"] == "neutral").sum()))
c4.metric("Negatif", int((f["sentiment_label"] == "negative").sum()))

# ── Volume & sentimen ───────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Volume per hari")
    if f["dt"].notna().any():
        vol = f.dropna(subset=["dt"]).set_index("dt").resample("D").size()
        st.line_chart(vol)
    else:
        st.info("Tanggal tidak tersedia.")
with right:
    st.subheader("Distribusi sentimen")
    st.bar_chart(f["sentiment_label"].value_counts())

# ── Tren sentimen per topik ─────────────────────────────────────
_SENT_VAL = {"positive": 1, "neutral": 0, "negative": -1}


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


st.subheader("📈 Tren sentimen per topik")
topics_df = explode_topics(f)
if topics_df.empty or topics_df["dt"].isna().all():
    st.info("Belum ada data topik/tanggal. Pastikan `keywords` terisi di config.yaml.")
else:
    topics_df = topics_df.dropna(subset=["dt"])
    all_topics = sorted(topics_df["topic"].unique().tolist())
    colf1, colf2 = st.columns([3, 1])
    with colf1:
        sel_topics = st.multiselect("Topik", all_topics, default=all_topics[:5])
    with colf2:
        freq = st.selectbox("Granularitas", ["Harian", "Mingguan"], index=0)
    rule = "D" if freq == "Harian" else "W"

    tf = topics_df[topics_df["topic"].isin(sel_topics)].copy()
    if tf.empty:
        st.info("Pilih minimal satu topik.")
    else:
        tf["val"] = tf["sentiment_label"].map(_SENT_VAL).fillna(0)

        # Indeks sentimen bersih (-1..1) per topik dari waktu ke waktu
        st.markdown("**Indeks sentimen bersih** (1 = sangat positif, −1 = sangat negatif)")
        net = (tf.set_index("dt").groupby("topic")["val"]
                 .resample(rule).mean().reset_index())
        if not net.empty:
            st.line_chart(net.pivot(index="dt", columns="topic", values="val"))

        # Volume perbincangan per topik
        st.markdown("**Volume perbincangan per topik**")
        vol = (tf.set_index("dt").groupby("topic")["val"]
                 .resample(rule).size().reset_index(name="jumlah"))
        st.area_chart(vol.pivot(index="dt", columns="topic", values="jumlah").fillna(0))

        # Rincian sentimen per topik (tabel)
        st.markdown("**Rincian sentimen per topik**")
        pivot = (tf.groupby(["topic", "sentiment_label"]).size()
                   .unstack(fill_value=0))
        for c in ("positive", "neutral", "negative"):
            if c not in pivot.columns:
                pivot[c] = 0
        pivot["total"] = pivot[["positive", "neutral", "negative"]].sum(axis=1)
        pivot["net %"] = ((pivot["positive"] - pivot["negative"]) /
                          pivot["total"].replace(0, 1) * 100).round(1)
        st.dataframe(pivot[["positive", "neutral", "negative", "total", "net %"]]
                     .sort_values("total", ascending=False), use_container_width=True)

# ── Sumber/aktor teratas ────────────────────────────────────────
st.subheader("Sumber / aktor teratas")
st.dataframe(
    f["source"].value_counts().head(15).rename_axis("source").reset_index(name="jumlah"),
    use_container_width=True,
)

# ── SNA ─────────────────────────────────────────────────────────
st.header("🕸️ Social Network Analysis")
plat_for_sna = "twitter" if "twitter" in sel_plat else (sel_plat[0] if sel_plat else None)
G = sna.build_graph(_db_path(), platform=plat_for_sna)

if G.number_of_nodes() == 0:
    st.info("Belum ada data interaksi (jalankan collector X untuk mengisi graf).")
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
        st.subheader("Paling berpengaruh (degree)")
        st.dataframe(pd.DataFrame(sna.top_actors(G, "degree", 15),
                                  columns=["aktor", "skor"]), use_container_width=True)
    with cb:
        st.subheader("Jembatan antar-klaster (betweenness)")
        bt = [(a, round(v, 4)) for a, v in sna.top_actors(G, "betweenness", 15)]
        st.dataframe(pd.DataFrame(bt, columns=["aktor", "skor"]), use_container_width=True)

    # Graf top-N (subgraph aktor terpenting) via graphviz, diwarnai per komunitas
    st.subheader("Peta jaringan (top 40 aktor)")
    top_nodes = [a for a, _ in sna.top_actors(G, "degree", 40)]
    sub = G.subgraph(top_nodes)
    palette = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
    dot = ["digraph { rankdir=LR; node [style=filled, fontsize=10];"]
    for n in sub.nodes():
        color = palette[parts.get(n, 0) % len(palette)]
        safe = n.replace('"', "")
        dot.append(f'"{safe}" [fillcolor="{color}"];')
    for u, v in sub.edges():
        dot.append(f'"{u.replace(chr(34),"")}" -> "{v.replace(chr(34),"")}";')
    dot.append("}")
    st.graphviz_chart("\n".join(dot))

st.caption("Data auto-refresh tiap 60 dtk. Atur sumber & kata kunci di config.yaml.")
