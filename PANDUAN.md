# 📖 Panduan Pemakaian

Panduan lengkap menjalankan sistem monitoring sosial & berita ini — dari nol
sampai baca hasil di dashboard.

## Daftar isi
1. [Gambaran alur](#1-gambaran-alur)
2. [Instalasi](#2-instalasi)
3. [Menjalankan](#3-menjalankan)
4. [Mengatur kata kunci & sumber](#4-mengatur-kata-kunci--sumber)
5. [Mengaktifkan X / Twitter](#5-mengaktifkan-x--twitter)
6. [Mengaktifkan Instagram](#6-mengaktifkan-instagram)
7. [Mengaktifkan Facebook](#7-mengaktifkan-facebook)
8. [Membaca dashboard](#8-membaca-dashboard)
9. [Mengganti mesin sentimen](#9-mengganti-mesin-sentimen)
10. [Menjalankan dengan Docker](#10-menjalankan-dengan-docker)
11. [Alur kerja harian yang disarankan](#11-alur-kerja-harian-yang-disarankan)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Gambaran alur

```
config.yaml  →  Collector (berita/X/IG/FB)  →  data/monitoring.db  →  Analisis (sentimen + SNA)  →  Dashboard
```

- **Collector** mengambil data sesuai kata kunci di `config.yaml`.
- Semua disimpan ke satu database SQLite `data/monitoring.db`.
- **Sentimen** (IndoBERT) & **SNA** (graf interaksi) dihitung dari data itu.
- **Dashboard** (Streamlit) menampilkan hasilnya.

Tiga cara menjalankan:
- `run_once.py` — satu siklus (uji cepat).
- `scheduler.py` — monitoring berkelanjutan (otomatis tiap N menit).
- `streamlit run dashboard/app.py` — buka dashboard.

---

## 2. Instalasi

Pilih **salah satu** (keduanya mandiri, tak bergantung Python sistem):

### Cara A — uv (native, disarankan)
```powershell
uv python install 3.12               # unduh Python 3.12 mandiri (sekali saja)
uv venv --python 3.12                # buat .venv
uv pip install -r requirements.txt   # install semua paket
```
Atau cukup jalankan `.\setup.ps1` (otomatis melakukan semuanya).

Belum punya uv? `pip install uv` atau `irm https://astral.sh/uv/install.ps1 | iex`.

### Cara B — Docker
Lihat [bagian 10](#10-menjalankan-dengan-docker).

> **Aktifkan venv dulu** sebelum menjalankan perintah `python ...` di bawah:
> ```powershell
> .\.venv\Scripts\Activate.ps1        # Windows
> # source .venv/bin/activate         # Linux/Mac
> ```
> Tanda aktif: ada `(.venv)` di awal baris terminal.

---

## 3. Menjalankan

**Uji sekali (berita langsung jalan tanpa kredensial):**
```powershell
python run_once.py
```
Contoh keluaran:
```
[news/rss] 5 feed...
  -> 8 cocok, 8 baru disimpan
[sentiment] mesin=indobert, 8 dokumen...
Selesai. Dokumen baru: 8 | dianalisis: 8
Statistik DB: {'total': 8, 'by_platform': {'news': 8}, ...}
```

**Monitoring berkelanjutan (biarkan jalan di background):**
```powershell
python scheduler.py     # tarik otomatis tiap N menit; Ctrl+C untuk berhenti
```

**Buka dashboard (terminal terpisah, venv aktif):**
```powershell
streamlit run dashboard/app.py
```
Buka browser ke http://localhost:8501.

---

## 4. Mengatur kata kunci & sumber

Semua diatur di **`config.yaml`** — tak perlu menyentuh kode.

**Kata kunci yang dipantau:**
```yaml
keywords:
  - "ibu kota nusantara"
  - "ikn"
  - "pemilu"
  - "subsidi bbm"
```
> Frasa berspasi otomatis dicocokkan utuh. Kata pendek (mis. `ikn`) memakai
> batas-kata, jadi tidak salah cocok ke "detik**N**ews".

**Sumber berita (RSS):** tambah/kurangi feed di `news.rss_feeds`.
```yaml
news:
  rss_feeds:
    - "https://news.detik.com/berita/rss"
    - "https://www.antaranews.com/rss/terkini.xml"
  fetch_full_text: true   # true = ambil isi artikel penuh; false = ringkasan RSS saja
  gdelt:
    enabled: true         # pencarian berita global via GDELT (gratis)
    timespan: "1d"
```

**Jadwal tarik:**
```yaml
schedule:
  news_minutes: 30        # berita tiap 30 menit
  x_minutes: 60           # X tiap 60 menit
  ig_minutes: 120         # Instagram tiap 120 menit
```

---

## 5. Mengaktifkan X / Twitter

Ada 2 jalur, diatur di `config.yaml → x.method`:
- `twikit` — gratis (login akun)
- `service` — berbayar (Apify), stabil
- `auto` — coba twikit dulu, fallback ke Apify bila kosong/limit

### Jalur gratis (twikit)
1. Install: `uv pip install twikit` (sudah termasuk di requirements).
2. Siapkan akun **cadangan** (jangan akun utama — berisiko dikunci):
   ```powershell
   copy secrets\x_accounts.yaml.example secrets\x_accounts.yaml
   ```
3. Isi `secrets/x_accounts.yaml`:
   ```yaml
   accounts:
     - username: "akun_cadangan_1"
       email: "email1@example.com"
       password: "password1"
       proxy: ""          # opsional: "http://user:pass@host:port"
   ```
4. Set query & aktifkan di `config.yaml`:
   ```yaml
   x:
     enabled: true
     method: "auto"
     search_queries: ["ikn", "subsidi bbm"]
     tweets_per_query: 50
   ```

> **Tips anti-limit:** makin banyak akun + proxy residensial, makin tahan.
> Sesi login otomatis di-cache di `data/x_cookies/` supaya tidak login berulang.

### Jalur berbayar (Apify)
```powershell
setx APIFY_TOKEN "apify_xxxxx"        # Windows (buka terminal baru setelah ini)
# export APIFY_TOKEN="apify_xxxxx"    # Linux/Mac
```
Lalu di `config.yaml → x.method: "service"` atau `"auto"`.

---

## 6. Mengaktifkan Instagram

1. Install: `uv pip install instaloader` (sudah di requirements).
2. Kredensial (akun cadangan):
   ```powershell
   copy secrets\ig_accounts.yaml.example secrets\ig_accounts.yaml
   ```
3. Aktifkan di `config.yaml`:
   ```yaml
   instagram:
     enabled: true
     hashtags: ["ikn", "pemilu"]
     posts_per_tag: 30
   ```
> IG sangat agresif membatasi. Pakai `posts_per_tag` kecil, jeda besar
> (`min_delay_sec`/`max_delay_sec`), dan akun cadangan.

---

## 7. Mengaktifkan Facebook

Scraping FB langsung tidak bisa diandalkan. Jalur sah = **impor** dari
Meta Content Library (butuh akses peneliti):
1. Ekspor data dari Content Library ke CSV/JSON.
2. Simpan filenya, mis. `data/fb_export.csv`.
3. Di `config.yaml`:
   ```yaml
   facebook:
     enabled: true
     import_file: "data/fb_export.csv"
   ```
Importer memetakan kolom umum (`account_name`, `message`, `likes`, dst) otomatis.

---

## 8. Membaca dashboard

Buka http://localhost:8501 setelah `streamlit run dashboard/app.py`.

- **Ringkasan** — total dokumen + jumlah positif/netral/negatif.
- **Volume per hari** — grafik jumlah perbincangan dari waktu ke waktu.
- **Distribusi sentimen** — proporsi positif/netral/negatif.
- **Tren sentimen per topik** — inti gaya Drone Emprit:
  - *Indeks sentimen bersih* (1 = sangat positif, −1 = sangat negatif) per kata kunci.
  - *Volume perbincangan* per topik.
  - *Tabel net %* per topik.
- **Social Network Analysis** (muncul bila ada data interaksi X/IG):
  - *Paling berpengaruh (degree)* — akun paling banyak berinteraksi.
  - *Jembatan antar-klaster (betweenness)* — akun penghubung antar kelompok.
  - *Peta jaringan* — graf diwarnai per komunitas.

Filter platform & sentimen ada di sidebar kiri. Data auto-refresh tiap 60 detik.

Analisis SNA juga bisa dilihat di terminal:
```powershell
python -m analysis.sna
```

---

## 9. Mengganti mesin sentimen

Di `config.yaml`:
```yaml
sentiment:
  engine: "indobert"   # akurat (default). Model ~500MB terunduh sekali.
  # engine: "lexicon"  # ringan, cepat, tanpa download — untuk uji cepat.
```

---

## 10. Menjalankan dengan Docker

Butuh Docker Desktop nyala.

```bash
docker compose up -d --build      # nyalakan scheduler + dashboard
docker compose logs -f scheduler  # pantau proses scraping
docker compose down               # matikan
```
- Dashboard: http://localhost:8501
- `config.yaml` & `secrets/` di-bind-mount → ubah kata kunci/kredensial **tanpa rebuild**, cukup restart: `docker compose restart`.
- Database tersimpan di `./data` (persist walau container dimatikan).

---

## 11. Alur kerja harian yang disarankan

1. Atur `keywords` & sumber di `config.yaml` sesuai topik yang sedang dipantau.
2. Jalankan `python scheduler.py` (atau `docker compose up -d`) dan biarkan.
3. Buka dashboard kapan saja untuk melihat tren & jaringan.
4. Untuk laporan/screenshot, filter platform/topik di sidebar dashboard.
5. Backup berkala: cukup salin file `data/monitoring.db`.

---

## 12. Troubleshooting

| Masalah | Sebab & solusi |
|---|---|
| Berita 0 padahal kata kunci ada | Belum ada artikel baru yang cocok saat itu — normal, coba lagi nanti atau tambah feed. |
| `[gdelt] 429 / timeout` | GDELT membatasi laju atau jaringan lambat. Sudah ada retry otomatis; aman diabaikan. |
| `[x/twikit] belum ada akun` | Isi `secrets/x_accounts.yaml`. |
| `[x/twikit] kena limit -> rotasi akun` | Wajar; tambah akun/proxy agar lebih tahan. |
| Akun X/IG kena suspend | Risiko jalur gratis. Pakai akun cadangan, atau pindah ke Apify (`method: service`). |
| Model IndoBERT lama diunduh | Hanya sekali (~500MB), lalu di-cache. Untuk uji cepat pakai `engine: lexicon`. |
| Dashboard "Database kosong" | Jalankan `python run_once.py` dulu agar ada data. |
| Bagian SNA kosong | Belum ada data interaksi — aktifkan collector X/IG. |
| `.venv` rusak setelah update Python | Pakai runtime uv/Docker (mandiri). Buat ulang: `.\setup.ps1`. |

---

Butuh bantuan lebih lanjut? Lihat [README.md](README.md) untuk ringkasan arsitektur.
