"""Jalankan satu siklus pengumpulan + analisis sentimen (sekali jalan).

Pakai untuk uji cepat / riset sekali jalan:
    python run_once.py
"""
from __future__ import annotations

import yaml

from core.storage import Storage
from collectors import news_rss, news_gdelt, x_collect, instagram, facebook
from analysis.sentiment import SentimentEngine


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_news(cfg: dict, store: Storage) -> int:
    kw = cfg.get("keywords", [])
    lang = cfg.get("language", "")
    news = cfg.get("news", {})
    total_new = 0

    feeds = news.get("rss_feeds", [])
    if feeds:
        print(f"[news/rss] {len(feeds)} feed...")
        docs = news_rss.collect(feeds, kw, lang, news.get("fetch_full_text", False))
        n = store.save_documents(docs)
        print(f"  -> {len(docs)} cocok, {n} baru disimpan")
        total_new += n

    gd = news.get("gdelt", {})
    if gd.get("enabled"):
        print("[news/gdelt] query...")
        docs = news_gdelt.collect(kw, lang, gd.get("timespan", "1d"), gd.get("max_records", 75))
        n = store.save_documents(docs)
        print(f"  -> {len(docs)} artikel, {n} baru disimpan")
        total_new += n

    return total_new


def collect_x(cfg: dict, store: Storage) -> int:
    xcfg = cfg.get("x", {})
    if not xcfg.get("enabled"):
        return 0
    print("[x/twitter] kumpulkan...")
    docs, edges = x_collect.collect(xcfg)
    n = store.save_documents(docs)
    ne = store.save_interactions(edges)
    print(f"  -> {len(docs)} tweet ({n} baru), {ne} edge interaksi disimpan")
    return n


def collect_instagram(cfg: dict, store: Storage) -> int:
    igcfg = cfg.get("instagram", {})
    if not igcfg.get("enabled"):
        return 0
    print("[instagram] kumpulkan...")
    docs, edges = instagram.collect(igcfg)
    n = store.save_documents(docs)
    ne = store.save_interactions(edges)
    print(f"  -> {len(docs)} post ({n} baru), {ne} edge disimpan")
    return n


def collect_facebook(cfg: dict, store: Storage) -> int:
    fbcfg = cfg.get("facebook", {})
    if not fbcfg.get("enabled"):
        return 0
    print("[facebook] impor...")
    docs, edges = facebook.collect(fbcfg)
    n = store.save_documents(docs)
    ne = store.save_interactions(edges)
    print(f"  -> {len(docs)} post ({n} baru), {ne} edge disimpan")
    return n


def analyze_sentiment(cfg: dict, store: Storage) -> int:
    engine = SentimentEngine(cfg.get("sentiment", {}))
    pending = store.docs_without_sentiment(limit=2000)
    if not pending:
        return 0
    print(f"[sentiment] mesin={engine.engine}, {len(pending)} dokumen...")
    texts = [f"{r.get('title','')} {r.get('content','')}".strip() for r in pending]
    results = engine.predict_batch(texts)            # batch = cepat utk IndoBERT
    for row, (label, score) in zip(pending, results):
        store.update_sentiment(row["doc_id"], label, score)
    return len(pending)


def main():
    cfg = load_config()
    store = Storage(cfg["storage"]["db_path"])

    print("=" * 50)
    new_docs = collect_news(cfg, store)
    new_docs += collect_x(cfg, store)
    new_docs += collect_instagram(cfg, store)
    new_docs += collect_facebook(cfg, store)
    analyzed = analyze_sentiment(cfg, store)

    print("=" * 50)
    print(f"Selesai. Dokumen baru: {new_docs} | dianalisis: {analyzed}")
    print("Statistik DB:", store.stats())


if __name__ == "__main__":
    main()
