# 📡 Scrapping ver2.1 — Monitoring Sosial & Berita

Pipeline pengumpulan data **multi-platform** (Berita, X/Twitter, Instagram, Facebook)
untuk **analisis sentimen** dan **Social Network Analysis (SNA)** — terinspirasi
pendekatan [Drone Emprit](https://pers.droneemprit.id/). Dirancang untuk
**monitoring berkelanjutan**: kumpulkan → simpan → analisis → visualisasikan.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-aktif-success)

---

## ✨ Fitur

| Kemampuan | Detail |
|---|---|
| **Multi-sumber** | RSS media + GDELT (berita), X/Twitter, Instagram, Facebook |
| **Sentimen Bahasa Indonesia** | IndoBERT (default, akurat) + lexicon (fallback ringan) |
| **SNA** | Graf interaksi mention/retweet/quote → top aktor, betweenness, komunitas (Louvain) |
| **Anti rate-limit** | Rotasi akun, cache sesi, proxy, backoff (jalur gratis) + fallback layanan berbayar |
| **Monitoring berkelanjutan** | Scheduler per-platform dengan interval terpisah |
| **Dashboard** | Streamlit: volume, distribusi sentimen, **tren sentimen per topik**, peta jaringan |
| **Skema seragam** | Semua platform → satu tabel; SQLite, mudah migrasi ke PostgreSQL |

## 🏗️ Arsitektur

```
config.yaml ─┐
             ▼
   ┌─────────────────────────────────────┐   scheduler (APScheduler)
   │ Collectors: News · X · IG · Facebook │   jalan berkala
   └───────────────────┬─────────────────┘
                       ▼
            Storage (SQLite → PostgreSQL)
            documents  +  interactions
                       ▼
   ┌─────────────────────────────────────┐
   │ Analysis: Sentimen (IndoBERT) · SNA  │
   └───────────────────┬─────────────────┘
                       ▼
            Dashboard (Streamlit)
```

## 🚀 Mulai cepat

```bash
git clone https://github.com/uvukukiland/scrappingll_ver2.1.git
cd scrappingll_ver2.1
pip install -r requirements.txt

python run_once.py                  # 1x siklus (Berita langsung jalan, tanpa kredensial)
python scheduler.py                 # monitoring berkelanjutan (Ctrl+C berhenti)
streamlit run dashboard/app.py      # buka dashboard
```

> **Berita** bekerja tanpa setup apa pun. Platform sosial perlu kredensial
> (lihat di bawah) dan diaktifkan via `enabled: true` di `config.yaml`.

## ⚙️ Konfigurasi

Semua diatur di [`config.yaml`](config.yaml) — kode tak perlu disentuh:
- `keywords` — topik yang dipantau (filter berita + query GDELT)
- `news.rss_feeds` — daftar feed media
- `x` / `instagram` / `facebook` — sumber sosial (`enabled`, query, jadwal)
- `sentiment.engine` — `indobert` (akurat) atau `lexicon` (ringan)
- `schedule.*_minutes` — interval tarik per platform

## 🔌 Mengaktifkan platform sosial

<details>
<summary><b>X / Twitter</b> — gratis (twikit) atau berbayar (Apify)</summary>

Atur `config.yaml → x.method`: `twikit` | `service` | `auto`.

```bash
# Gratis (login akun, rawan limit pada volume besar)
pip install twikit
cp secrets/x_accounts.yaml.example secrets/x_accounts.yaml   # isi akun + proxy

# Berbayar (stabil, anti-bot ditangani vendor)
setx APIFY_TOKEN "apify_xxx"        # Windows;  export APIFY_TOKEN=... di Linux/Mac
```
`auto` = coba twikit dulu, fallback ke Apify saat kosong/limit.
</details>

<details>
<summary><b>Instagram</b> — instaloader (konten publik)</summary>

```bash
pip install instaloader
cp secrets/ig_accounts.yaml.example secrets/ig_accounts.yaml   # akun cadangan
```
IG sangat membatasi: pakai akun cadangan, `posts_per_tag` kecil, jeda besar.
</details>

<details>
<summary><b>Facebook</b> — importer Meta Content Library</summary>

Scraping FB langsung tidak viable (ToS + anti-bot). Jalur sah: ekspor data dari
**Meta Content Library** (akses peneliti), simpan CSV/JSON, lalu set
`config.yaml → facebook.import_file` & `enabled: true`. Importer memetakan kolom
umum (`account_name`, `message`, `likes`, dst) otomatis.
</details>

## 🧠 Sentimen

Default **IndoBERT** (`mdhugol/indonesia-bert-sentiment-classification`) — akurat,
diproses **batch**, model (~500 MB) terunduh sekali otomatis. Bila
`transformers`/`torch` tak tersedia, otomatis fallback ke **lexicon** (nol-setup).

## 🕸️ SNA & Dashboard

- **SNA** ([`analysis/sna.py`](analysis/sna.py)): graf terarah dari tabel
  `interactions` → aktor paling berpengaruh (degree), penghubung antar-klaster
  (betweenness), dan deteksi komunitas (Louvain).
- **Dashboard** ([`dashboard/app.py`](dashboard/app.py)): volume per hari,
  distribusi sentimen, **tren & indeks sentimen bersih per topik**, dan peta
  jaringan diwarnai per komunitas.

## 📂 Struktur

```
config.yaml            # kata kunci, sumber, jadwal
core/        models.py · storage.py            # skema seragam + SQLite
collectors/  base.py · news_rss · news_gdelt   # berita
             x_twikit · x_service · x_collect   # X (gratis/berbayar/auto)
             instagram · facebook               # IG + importer FB
analysis/    sentiment.py · sna.py             # sentimen + SNA
dashboard/   app.py                            # Streamlit
run_once.py · scheduler.py                     # runner
secrets/     *.yaml.example                    # template kredensial (gitignored)
```

## 🗺️ Roadmap

- [x] Fase 1 — Fondasi + Berita (RSS + GDELT)
- [x] Fase 2 — X/Twitter (twikit + Apify)
- [x] Fase 3 — SNA + Dashboard
- [x] Fase 4 — Instagram + Facebook
- [ ] Migrasi PostgreSQL untuk skala besar
- [ ] Deteksi bot / klaster buzzer
- [ ] Alert otomatis (lonjakan negatif)

## ⚖️ Etika & Legal

Alat ini ditujukan untuk **riset, jurnalistik, dan analisis kepentingan publik**.
Tanggung jawab pengguna:
- Patuhi **Terms of Service** tiap platform dan hukum yang berlaku (mis. UU PDP).
- Hormati data pribadi; **anonimkan** saat menyajikan, hindari menarget individu.
- Gunakan jeda yang wajar; jangan membebani server sumber.
- Scraping platform sosial bisa melanggar ToS mereka — risiko (mis. blokir akun)
  ditanggung pengguna. Penulis tidak bertanggung jawab atas penyalahgunaan.

## 📄 Lisensi

[MIT](LICENSE) © 2026 uvukukiland
