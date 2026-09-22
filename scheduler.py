"""Monitoring berkelanjutan (Fase 1).

Menjalankan siklus pengumpulan berita + sentimen tiap N menit (config.yaml).
    python scheduler.py
Hentikan dengan Ctrl+C.
"""
from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

import os
from datetime import datetime

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

    def _muat_ulang():
        """Baca ulang config tiap siklus: ubah kata kunci/batas tanpa restart.

        Config rusak (mis. sedang diedit) tidak boleh mematikan scheduler —
        siklus itu memakai config lama dan mencoba lagi nanti.
        """
        nonlocal cfg
        try:
            cfg = load_config()
        except Exception as e:
            print(f"[scheduler] config gagal dibaca ({e}) -> pakai yang lama")
        return cfg

    def news_cycle():
        _stamp("NEWS")
        try:
            cfg = _muat_ulang()
            collect_news(cfg, store)
            collect_facebook(cfg, store)     # impor file (idempoten, murah)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error news: {e}")

    def ig_cycle():
        _stamp("INSTAGRAM")
        try:
            cfg = _muat_ulang()
            collect_instagram(cfg, store)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error ig: {e}")

    def x_cycle():
        _stamp("X/TWITTER")
        try:
            cfg = _muat_ulang()
            collect_x(cfg, store)
            analyze_sentiment(cfg, store)
            print("stats:", store.stats())
        except Exception as e:
            print(f"[scheduler] error x: {e}")

    def laporan_harian():
        """Buat laporan harian; kirim email HANYA bila kamu menyalakannya."""
        _stamp("LAPORAN")
        cfg = _muat_ulang()
        lap = cfg.get("laporan", {}) or {}
        try:
            from datetime import timedelta
            from tools.laporan import buat
            hari = int(lap.get("hari_terakhir", 1))
            akhir = datetime.now()
            folder = buat(mulai=(akhir - timedelta(days=hari)).strftime("%Y-%m-%d"),
                          akhir=akhir.strftime("%Y-%m-%d"),
                          entitas=lap.get("entitas") or None)
        except Exception as e:
            print(f"[scheduler] laporan gagal dibuat: {e}")
            return

        surel = lap.get("email", {}) or {}
        if not surel.get("enabled"):
            print("[scheduler] email dimatikan — laporan hanya disimpan ke folder.")
            return
        try:
            from tools.kirim_email import kirim, EmailTidakSiap
            ringkas = ""
            berkas_md = os.path.join(folder, "ringkasan.md")
            if os.path.isfile(berkas_md):
                ringkas = open(berkas_md, encoding="utf-8").read()
            hasil = kirim(folder, surel, ringkasan=ringkas)
            print(f"[scheduler] laporan dikirim ke {', '.join(hasil['penerima'])} "
                  f"({len(hasil['lampiran'])} lampiran)")
        except EmailTidakSiap as e:
            print(f"[scheduler] email dilewati: {e}")
        except Exception as e:
            print(f"[scheduler] gagal mengirim email: {e}")

    sched = BlockingScheduler(timezone="Asia/Jakarta")
    sched.add_job(news_cycle, "interval", minutes=news_min)
    if cfg.get("x", {}).get("enabled"):
        sched.add_job(x_cycle, "interval", minutes=x_min)
    if cfg.get("instagram", {}).get("enabled"):
        sched.add_job(ig_cycle, "interval", minutes=ig_min)

    # Laporan harian: cron pada jam yang disetel. Mengubah jamnya perlu restart
    # scheduler, sama seperti schedule.*_minutes.
    lap_cfg = cfg.get("laporan", {}) or {}
    jam_lap = ""
    if lap_cfg.get("enabled"):
        jam_lap = str(lap_cfg.get("jam", "07:00"))
        try:
            jam, menit = (int(x) for x in jam_lap.split(":")[:2])
        except ValueError:
            jam, menit, jam_lap = 7, 0, "07:00 (format jam tak dikenali)"
        sched.add_job(laporan_harian, "cron", hour=jam, minute=menit)

    news_cycle()                       # jalankan sekali di awal
    if cfg.get("x", {}).get("enabled"):
        x_cycle()
    if cfg.get("instagram", {}).get("enabled"):
        ig_cycle()
    jadwal = f"news/{news_min}m, x/{x_min}m, ig/{ig_min}m"
    if jam_lap:
        jadwal += f", laporan harian {jam_lap}"
    print(f"\n[scheduler] aktif: {jadwal}. Ctrl+C berhenti.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] berhenti.")


if __name__ == "__main__":
    main()
