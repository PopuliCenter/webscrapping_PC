"""Laporan monitoring otomatis: ringkasan, Excel, grafik, dan halaman HTML.

Menghasilkan satu folder siap kirim ke atasan/klien:

    laporan/2026-09-22/
      ringkasan.md      ringkasan eksekutif (teks, bisa disalin ke email/WA)
      laporan.xlsx      data mentah + tabel: Ringkasan, Share of Voice, Media,
                        Sentimen harian, Dokumen
      laporan.html      versi siap cetak (Ctrl+P -> simpan sebagai PDF)
      *.png             grafik volume harian, sebaran sentimen, share of voice

Semua angka mengikuti aturan yang sama dengan dashboard: berita sindikasi
dihitung sekali, dan kata dihitung per dokumen.

Jalankan:
    python tools/laporan.py                      # 7 hari terakhir
    python tools/laporan.py --hari 30
    python tools/laporan.py --mulai 2026-09-01 --akhir 2026-09-22
    python tools/laporan.py --entitas AS Iran "Selat Hormuz"
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WARNA = {"positive": "#2e9e5b", "neutral": "#8a8f98", "negative": "#d64545"}


def _grafik_volume(df, tujuan: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    harian = (df.dropna(subset=["dt"]).set_index("dt")
              .tz_convert("Asia/Jakarta").resample("D").size())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(harian.index, harian.values, marker="o", color="#3b7dd8")
    ax.fill_between(harian.index, harian.values, alpha=0.15, color="#3b7dd8")
    ax.set_title("Volume pemberitaan per hari")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(tujuan, dpi=120)
    plt.close(fig)
    return tujuan


def _grafik_sentimen(df, tujuan: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = df["sentiment_label"].value_counts()
    urut = [k for k in ("positive", "neutral", "negative") if k in n]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([k for k in urut], [n[k] for k in urut],
           color=[WARNA[k] for k in urut])
    ax.set_title("Sebaran sentimen")
    for i, k in enumerate(urut):
        ax.text(i, n[k], str(n[k]), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(tujuan, dpi=120)
    plt.close(fig)
    return tujuan


def _grafik_sov(tabel, tujuan: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tabel.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.45 * len(t))))
    ax.barh(t["entitas"], t["dokumen"], color="#3b7dd8")
    for i, (n, s) in enumerate(zip(t["dokumen"], t["share_%"])):
        ax.text(n, i, f" {n} ({s}%)", va="center")
    ax.set_title("Share of Voice (jumlah berita)")
    fig.tight_layout()
    fig.savefig(tujuan, dpi=120)
    plt.close(fig)
    return tujuan


def _ringkasan_teks(df, tabel, lonj, mulai, akhir, entitas, disaring: int = 0,
                    riwayat_hari: int = 0) -> str:
    n = len(df)
    s = df["sentiment_label"].value_counts()
    pos, net, neg = int(s.get("positive", 0)), int(s.get("neutral", 0)), int(s.get("negative", 0))
    indeks = round((pos - neg) / n, 3) if n else 0
    media = df["source"].value_counts()

    baris = [f"# Laporan monitoring {mulai} s/d {akhir}", "",
             f"**{n} berita** dari **{df['source'].nunique()} media**.",
             f"Nada keseluruhan: {pos} positif · {net} netral · {neg} negatif "
             f"(indeks {indeks:+.3f}, rentang -1 s/d +1).", ""]

    if lonj and lonj.get("lonjakan"):
        baris.append(f"> ⚠️ **Lonjakan volume** pada {lonj['tanggal']}: "
                     f"{lonj['jumlah']} berita, {lonj['rasio']}x rata-rata "
                     f"({lonj['rata_rata']}/hari).")
        if riwayat_hari and riwayat_hari < 7:
            # Pemantauan yang baru jalan beberapa hari SELALU terlihat melonjak:
            # hari-hari awal sepi karena datanya belum terkumpul, bukan karena
            # isunya kecil. Ini disebut agar laporan tidak salah dibaca.
            baris.append(f"> Hati-hati menafsirkan: pemantauan baru berjalan "
                         f"{riwayat_hari} hari, jadi kenaikan ini bisa berasal "
                         f"dari data yang baru terkumpul.")
        baris.append("")
    elif lonj and lonj.get("catatan"):
        baris.append(f"> Deteksi lonjakan belum berlaku: {lonj['catatan']}.\n")

    baris += ["## Share of Voice", "",
              "| Pihak/isu | Berita | Share | Positif | Netral | Negatif | Indeks nada |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in tabel.iterrows():
        baris.append(f"| {r['entitas']} | {r['dokumen']} | {r['share_%']}% | "
                     f"{r['positif']} | {r['netral']} | {r['negatif']} | "
                     f"{r['indeks_nada']:+.3f} |")
    baris += ["", "_Satu berita bisa menyebut lebih dari satu pihak, jadi jumlah "
              "share bisa melebihi 100%._", ""]

    baris += ["## Media paling aktif", ""]
    for m, jml in media.head(8).items():
        from analysis.media import tier_media
        baris.append(f"- {m}: {jml} berita (tier {tier_media(m)})")

    from analysis.media import ringkas_tier
    tier = ringkas_tier(df)
    if tier:
        baris += ["", "Sebaran mutu sumber: " +
                  " · ".join(f"tier {t['tier']}: {t['dokumen']} ({t['share_%']}%)"
                             for t in tier)]

    negatif = df[df["sentiment_label"] == "negative"]
    if "bagian_negatif" in negatif.columns:
        negatif = negatif.sort_values("bagian_negatif", ascending=False)
    if len(negatif):
        baris += ["", "## Berita paling negatif", ""]
        for _, r in negatif.head(5).iterrows():
            kutip = str(r.get("kutipan_negatif") or "")[:160]
            baris.append(f"- **{(r['title'] or '')[:90]}** ({r['source']})")
            if kutip:
                baris.append(f"  > {kutip}")

    from analysis.entitas import dari_dokumen
    kutip, ent = dari_dokumen(df.head(150))
    if ent:
        baris += ["", "## Tokoh & lembaga paling sering disebut", ""]
        for nama, jml in ent[:10]:
            baris.append(f"- {nama}: {jml}x")
    if kutip:
        from collections import Counter as _C
        bicara = _C(k["pembicara"] for k in kutip)
        baris += ["", "## Kutipan penting", "",
                  "Pembicara terbanyak: " +
                  ", ".join(f"{n} ({j})" for n, j in bicara.most_common(5)), ""]
        for k in kutip[:6]:
            baris.append(f"- **{k['pembicara']}** ({k['source']}): "
                         f"\"{k['kutipan'][:180]}\"")

    catatan = [f"Dibuat otomatis {datetime.now():%Y-%m-%d %H:%M}",
               "berita sindikasi dihitung sekali",
               f"entitas: {', '.join(entitas)}"]
    if disaring:
        catatan.insert(1, f"{disaring} berita disaring karena hanya menyinggung "
                          f"kata kunci sekilas")
    baris += ["", "---", "", "_" + " · ".join(catatan) + "_"]
    return "\n".join(baris)


def _tulis_excel(path, df, tabel, tren_tabel):
    import pandas as pd
    kolom = [k for k in ("published_at", "source", "title", "url", "sentiment_label",
                         "sentiment_score", "bagian_negatif", "bagian_total",
                         "kutipan_negatif", "emotion_label") if k in df.columns]
    harian = (df.dropna(subset=["dt"]).set_index("dt").tz_convert("Asia/Jakarta")
              .groupby([lambda x: x.date(), "sentiment_label"]).size()
              .unstack(fill_value=0))
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        tabel.to_excel(w, sheet_name="Share of Voice", index=False)
        harian.to_excel(w, sheet_name="Sentimen harian")
        (df["source"].value_counts().rename_axis("media")
         .reset_index(name="berita")).to_excel(w, sheet_name="Media", index=False)
        if not tren_tabel.empty:
            tren_tabel.to_excel(w, sheet_name="Tren harian")
        from analysis.entitas import dari_dokumen as _dari
        kutip, ent = _dari(df.head(150))
        if ent:
            pd.DataFrame(ent, columns=["nama", "sebutan"]).to_excel(
                w, sheet_name="Entitas", index=False)
        if kutip:
            pd.DataFrame(kutip)[["pembicara", "kutipan", "source", "sentimen",
                                 "judul"]].to_excel(w, sheet_name="Kutipan", index=False)
        df[kolom].to_excel(w, sheet_name="Dokumen", index=False)
    return path


def _tulis_html(path, ringkasan_md: str, gambar: list):
    """HTML sederhana siap cetak — Ctrl+P lalu 'Save as PDF'."""
    import html
    isi = html.escape(ringkasan_md)
    img = "\n".join(f'<img src="{os.path.basename(g)}" style="max-width:100%">'
                    for g in gambar)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html lang="id"><meta charset="utf-8">
<title>Laporan monitoring</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;
padding:0 16px;line-height:1.55}} pre{{white-space:pre-wrap;font-family:inherit}}
img{{margin:12px 0;border:1px solid #ddd;border-radius:6px}}</style>
<pre>{isi}</pre>
{img}
</html>""")
    return path


def buat(db_path: str = "", mulai: str = "", akhir: str = "", entitas: list = None,
         keluar: str = "") -> str:
    import yaml
    from analysis import sov

    cfg = yaml.safe_load(open(os.path.join(PROJECT_DIR, "config.yaml"),
                              encoding="utf-8")) or {}
    db_path = db_path or os.path.join(PROJECT_DIR, cfg["storage"]["db_path"])
    entitas = [e for e in (entitas or cfg.get("keywords", [])) if str(e).strip()]

    df = sov.muat_dokumen(db_path, mulai, akhir)
    if df.empty:
        raise SystemExit("Tidak ada dokumen pada rentang itu.")
    # Buang berita yang hanya menyinggung kata kunci sekilas ("dolar AS" di
    # berita judi). Jumlahnya dilaporkan, bukan disembunyikan.
    df, disaring = sov.saring_relevan(df, entitas)
    if df.empty:
        raise SystemExit("Semua dokumen tersaring sebagai tidak relevan.")
    tabel = sov.hitung(df, entitas)
    tren_tabel = sov.tren(df, entitas)
    lonj = sov.lonjakan(df)

    folder = keluar or os.path.join(PROJECT_DIR, "laporan",
                                    datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)

    gambar = [_grafik_volume(df, os.path.join(folder, "volume.png")),
              _grafik_sentimen(df, os.path.join(folder, "sentimen.png"))]
    if not tabel.empty:
        gambar.append(_grafik_sov(tabel, os.path.join(folder, "share_of_voice.png")))

    import sqlite3 as _sq
    with _sq.connect(db_path) as _c:
        awal = _c.execute("SELECT MIN(collected_at) FROM documents").fetchone()[0]
    riwayat = 0
    if awal:
        try:
            riwayat = (datetime.now().astimezone()
                       - datetime.fromisoformat(awal)).days + 1
        except Exception:
            riwayat = 0
    md = _ringkasan_teks(df, tabel, lonj, mulai or str(df["dt"].min().date()),
                         akhir or str(df["dt"].max().date()), entitas,
                         disaring, riwayat)
    with open(os.path.join(folder, "ringkasan.md"), "w", encoding="utf-8") as f:
        f.write(md)
    _tulis_excel(os.path.join(folder, "laporan.xlsx"), df, tabel, tren_tabel)
    _tulis_html(os.path.join(folder, "laporan.html"), md, gambar)

    print(f"Laporan dibuat di: {folder}")
    for nama in sorted(os.listdir(folder)):
        ukuran = os.path.getsize(os.path.join(folder, nama)) / 1024
        print(f"  {nama:<22} {ukuran:>7.0f} KB")
    print(f"\n{md.splitlines()[2]}")
    return folder


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = sys.argv[1:]

    def opsi(nama, bawaan=None):
        return a[a.index(nama) + 1] if nama in a else bawaan

    ent = []
    if "--entitas" in a:
        for v in a[a.index("--entitas") + 1:]:
            if v.startswith("--"):
                break
            ent.append(v)
    hari = int(opsi("--hari", 7))
    buat(mulai=opsi("--mulai", (datetime.now() - timedelta(days=hari)).strftime("%Y-%m-%d")),
         akhir=opsi("--akhir", datetime.now().strftime("%Y-%m-%d")),
         entitas=ent)
