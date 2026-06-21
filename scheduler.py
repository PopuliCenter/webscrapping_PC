"""Monitoring berkelanjutan (Fase 1).

Menjalankan siklus pengumpulan berita + sentimen tiap N menit (config.yaml).
    python scheduler.py
Hentikan dengan Ctrl+C.
"""
from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from run_once import (load_config, collect_news, collect_x, collect_instagram,
                      collect_facebook, analyze_sentiment)
from core.storage import Storage


def main():
    cfg = load_config()
    store = Storage(cfg["storage"]["db_path"])
    sc = cfg.get("schedule", {})
    news_min = int(sc.get("news_minutes", 30))
    x_min = int(sc.get("x_minutes", 60))
    ig_min = int(sc.get("ig_minutes", 120))

    def _stamp(tag):
        from datetime import datetime
        print(f"\n--- {tag} {datetime.now():%Y-%m-%d %H:%M:%S} ---")

    def news_cycle():
        _stamp("NEWS")
        try:
            collect_news(cfg, store)
            collect_facebook(cfg, store)     # impor file (idempoten, murah)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error news: {e}")

    def ig_cycle():
        _stamp("INSTAGRAM")
        try:
            collect_instagram(cfg, store)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error ig: {e}")

    def x_cycle():
        _stamp("X/TWITTER")
        try:
            collect_x(cfg, store)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error x: {e}")

    sched = BlockingScheduler(timezone="Asia/Jakarta")
    sched.add_job(news_cycle, "interval", minutes=news_min)
    if cfg.get("x", {}).get("enabled"):
        sched.add_job(x_cycle, "interval", minutes=x_min)
    if cfg.get("instagram", {}).get("enabled"):
        sched.add_job(ig_cycle, "interval", minutes=ig_min)

    news_cycle()                       # jalankan sekali di awal
    if cfg.get("x", {}).get("enabled"):
        x_cycle()
    if cfg.get("instagram", {}).get("enabled"):
        ig_cycle()
    print(f"\n[scheduler] aktif: news/{news_min}m, x/{x_min}m, ig/{ig_min}m. Ctrl+C berhenti.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] berhenti.")


if __name__ == "__main__":
    main()
