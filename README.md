# 📡 Scrapping ver2.1 — Monitoring Sosial & Berita

Pipeline pengumpulan data **multi-platform** (Berita, X/Twitter, Instagram, Facebook)
untuk **analisis sentimen** dan **Social Network Analysis (SNA)** — terinspirasi
pendekatan [Drone Emprit](https://pers.droneemprit.id/). Dirancang untuk
**monitoring berkelanjutan**: kumpulkan → simpan → analisis → visualisasikan.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-aktif-success)

📖 **Baru pertama pakai? Baca [PANDUAN.md](PANDUAN.md)** — cara pakai lengkap langkah demi langkah.

---

## ✨ Fitur

| Kemampuan | Detail |
|---|---|
| **Multi-sumber** | RSS media + GDELT (berita), X/Twitter, Instagram, Facebook |
| **Preprocessing Bahasa Indonesia** | Hapus URL/mention/emoji/angka, normalisasi kata baku (slang→baku), stopword ID, **stemming Sastrawi**, deteksi near-duplicate |
| **Sentimen Bahasa Indonesia** | IndoBERT (default, akurat) + lexicon + **penilaian per-kata** (negasi & penguat, bisa diaudit) |
| **Klasifikasi ML** | TF-IDF + **SMOTE**, 6 algoritma (LogReg, Decision Tree, Random Forest, SVM, KNN, Naive Bayes), metrik **before vs after** |
| **Deteksi bot / buzzer** | Skor heuristik perilaku + deteksi **posting serentak** (coordinated behavior) |
| **Text mining** | Frekuensi kata, n-gram, kata khas per kelompok, **word cloud** |
| **SNA** | Graf interaksi mention/retweet/quote → top aktor, betweenness, komunitas (Louvain) |
| **Anti rate-limit** | Rotasi akun, cache sesi, proxy, backoff (jalur gratis) + fallback layanan berbayar |
| **Monitoring berkelanjutan** | Scheduler per-platform dengan interval terpisah |
| **Dashboard** | Streamlit 7 tab: ringkasan, tren per topik, word cloud, **heatmap**, jaringan, bot/buzzer, klasifikasi ML |
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

Proyek ini memakai runtime **mandiri** — Python-nya tidak bergantung pada Python
sistem, jadi aman meski kamu update/hapus Python di OS. Dua cara:

**Opsi A — uv (native, Python 3.12 mandiri di luar sistem):**
```powershell
git clone https://github.com/PopuliCenter/webscrapping_PC.git
cd webscrapping_PC
uv python install 3.12               # unduh Python mandiri (lepas dari sistem)
uv venv --python 3.12                # buat .venv
uv pip install -r requirements.txt   # install semua
.\.venv\Scripts\Activate.ps1         # aktifkan
python run_once.py                   # Berita langsung jalan, tanpa kredensial
streamlit run dashboard/app.py       # dashboard
```
`.python-version` sudah mem-pin ke 3.12 → `uv` otomatis pakai versi itu.
Belum punya uv? `pip install uv` atau `irm https://astral.sh/uv/install.ps1 | iex`.

**Opsi B — Docker (isolasi penuh: Python + OS ikut dalam image):**
```bash
docker compose up -d --build         # scheduler + dashboard
docker compose logs -f scheduler     # pantau
# dashboard: http://localhost:8501
```

> Hindari `python -m venv` biasa untuk pemakaian jangka panjang: venv itu meminjam
> Python sistem, jadi rusak bila Python sistem di-upgrade/dihapus.

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
             preprocess.py · textstats.py     # preprocessing ID + frekuensi kata
             lexicon_id.py · bot_detect.py    # skor per-kata + deteksi buzzer
             ml_classify.py                   # TF-IDF + SMOTE + 6 classifier
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
- [x] Fase 5 — Preprocessing ID, deteksi bot/buzzer, word cloud, heatmap, klasifikasi SMOTE
- [ ] Topic modeling (LDA / BERTopic) & analisis emosi
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

[MIT](LICENSE) © 2026 PopuliCenter
