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
12. [Analitik lanjutan (preprocessing, bot, word cloud, ML)](#12-analitik-lanjutan)
13. [Integrasi HuggingFace (dataset, banding model, emosi, Colab)](#13-integrasi-huggingface)
14. [Troubleshooting](#14-troubleshooting)

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

## 12. Analitik lanjutan

Dashboard punya **7 tab**. Empat di antaranya untuk analisis mendalam.

### a. Preprocessing teks (otomatis)
Setiap teks melewati tahapan ini sebelum dianalisis:

| Tahap | Contoh |
|---|---|
| Case folding | `Gakk SUKA` → `gakk suka` |
| Cleansing | buang URL, `@mention`, emoji, angka, tanda baca; `bangettt`→`banget` |
| Normalisasi kata baku | `gak`→`tidak`, `yg`→`yang`, `bgt`→`sangat` |
| Stopword removal | buang `yang`, `di`, `dan`, … (kata negasi **tidak** dibuang) |
| Stemming (Sastrawi) | `pembangunan` → `bangun` |
| Gabung negasi | `tidak bagus` → `tidak_bagus` (satu fitur tersendiri) |

> **Kenapa negasi dipertahankan?** Kalau `tidak` ikut dibuang sebagai stopword,
> `tidak suka` berubah jadi `suka` — maknanya terbalik. Karena itu kata negasi
> disimpan lalu digabung dengan kata sesudahnya. Matikan dengan
> `Preprocessor(merge_negation=False)` bila tak diinginkan.

**Kamus lokal yang bisa kamu edit** (folder `resources/`, berlaku langsung
tanpa melatih apa pun — cukup jalankan ulang program):

| File | Isi | Contoh efek |
|---|---|---|
| `kata_dasar_custom.txt` | kata dasar tambahan untuk Sastrawi | tambah `buzzer` → `pembuzzeran` jadi `buzzer` |
| `slang_baku.csv` | `slang,baku` | `nyampe,sampai` ; `cuy,` (baris kosong = kata dibuang) |
| `stopwords_custom.txt` | stopword tambahan | tambah `selengkapnya` agar noise RSS hilang |

Cek status kamus & model:
```powershell
python tools/setup_local_models.py --check
```

Lihat hasil tiap tahap untuk laporan metodologi:
```python
from analysis.preprocess import Preprocessor
p = Preprocessor()
for tahap, hasil in p.steps("teks kamu di sini").items():
    print(tahap, ":", hasil)
```
Tambah kamus sendiri: `Preprocessor(slang_file="kamus.csv", stopword_file="stopword.txt")`.
Matikan stemming lewat checkbox di sidebar bila terasa lambat.

### b. Tab ☁️ Teks & Word Cloud
- **Word cloud** per sentimen (Semua / positif / netral / negatif).
- **Kata paling sering** dan **frasa (bigram)** paling sering.
- **Kata khas negatif vs positif** — kata yang paling membedakan kedua kelompok.

### c. Tab 🔥 Heatmap
- **Jam × Hari** — kapan perbincangan memuncak. Aktivitas merata 24 jam = indikasi bot.
- **Topik × Sentimen** — topik mana yang paling negatif.

### d. Tab 🤖 Bot/Buzzer
Klik **Jalankan analisis bot**. Skor 0–1 dari lima sinyal perilaku:
frekuensi posting, rasio konten duplikat, sebaran jam aktif, pola username
(angka acak di belakang), dan rasio non-orisinal. Kategori: `rendah` /
`sedang` / `tinggi`.

Di bawahnya: **posting serentak** — teks identik yang disebar banyak akun
dalam waktu berdekatan. Ini sinyal terkuat kampanye terkoordinasi.

Menyaring bot dari analisis lain:
```python
from analysis.bot_detect import bot_actors
bots = bot_actors("data/monitoring.db", threshold=0.6)
```

### e. Tab 🧪 Klasifikasi ML (TF-IDF + SMOTE)
Membandingkan **6 algoritma** sebelum dan sesudah penyeimbangan data:
Logistic Regression, Decision Tree, Random Forest, SVM, K-Nearest Neighbors,
Naive Bayes.

Atur proporsi data uji, maksimum fitur TF-IDF, dan n-gram, lalu klik
**Latih & bandingkan model**. Hasil: tabel accuracy/precision/recall/F1
**before vs after SMOTE** (+ kolom Δ F1) dan **confusion matrix** per model.

Lewat terminal:
```powershell
python -m analysis.ml_classify
```

> **Catatan metodologi:** label latih diambil dari kolom sentimen di database
> (hasil IndoBERT). Untuk riset yang ketat, sebaiknya gunakan data yang
> dilabeli manual agar evaluasi tidak sirkular. SMOTE dilewati otomatis bila
> kelas terkecil punya terlalu sedikit sampel.

### f. Penilaian sentimen per-kata
Untuk menelusuri *mengapa* sebuah teks dinilai negatif:
```python
from analysis.lexicon_id import LexiconScorer
r = LexiconScorer().score("pelayanan tidak bagus, sangat kecewa")
print(r["label"], r["score"])
for d in r["details"]:
    print(d)   # kata, bobot_dasar, faktor, kontribusi, alasan
```
Menangani **negasi** (`tidak bagus` → membalik) dan **penguat** (`sangat`,
`sekali` → memperkuat). Pakai lexicon InSet bila ada:
`LexiconScorer(inset_dir="path/ke/inset")`.

### g. Model lokal & melatih ulang

**Simpan IndoBERT ke folder proyek** (sekali saja, ±500 MB) agar tidak
bergantung internet / cache HuggingFace:
```powershell
python tools/setup_local_models.py
```
Model masuk ke `models/indobert-sentiment/`. Folder `models/` sengaja
di-`.gitignore` karena besar — buat ulang dengan perintah yang sama.

**Dua cara "melatih" — beda sifatnya:**

| | Sastrawi | IndoBERT |
|---|---|---|
| Sifat | aturan + kamus kata dasar | jaringan saraf |
| Menambah kata baru | tulis di `resources/kata_dasar_custom.txt` | tidak cukup hanya kata |
| Perlu training? | **Tidak** — langsung berlaku | **Ya** — perlu contoh kalimat berlabel |
| Waktu | seketika | menit s/d jam |

IndoBERT tidak bisa "diajari kata" begitu saja: tokenizer-nya memecah kata
asing jadi sub-kata sehingga tetap terbaca. Yang perlu diajarkan adalah
**makna kata itu dalam kalimat**, lewat fine-tuning:

```powershell
# data latih CSV berkolom: text,label   (label: positive/neutral/negative)
python -m analysis.finetune_indobert --csv data/latih.csv --epochs 3
```
Hasil disimpan ke `models/indobert-sentiment-finetuned/` (model asli tidak
ditimpa). Pakai dengan mengubah `config.yaml`:
```yaml
sentiment:
  engine: "indobert"
  model_dir: "models/indobert-sentiment-finetuned"
```

> **Peringatan:** `--from-db` melatih memakai label yang dihasilkan IndoBERT
> sendiri, jadi sirkular — model hanya meniru dirinya. Untuk hasil yang sahih,
> gunakan `--csv` berisi data yang **dilabeli manual**. Sediakan idealnya
> ratusan contoh per kelas.

---

## 13. Integrasi HuggingFace

### a. Dataset berlabel manusia — memperbaiki masalah label sirkular

Melatih dari label keluaran IndoBERT sendiri itu sirkular. Dataset berikut
dilabeli **manusia**, jadi layak dipakai melatih:

| Kunci | Dataset | Baris | Kelas |
|---|---|---|---|
| `carant` | `carant-ai/indonesian_sentiment_dataset` | 1.030.393 | 3 (positive/neutral/negative) |
| `sepid` | `sepidmnorozy/Indonesian_sentiment` | 11.324 | 2 (biner) |

```python
from analysis.hf_datasets import load_labeled, simpan_csv
teks, label = load_labeled("carant", limit=20000, seimbang=True)
simpan_csv(teks, label, "data/latih.csv")
```
Data diambil **streaming** — tidak menarik sejuta baris. Lalu latih:
```powershell
python -m analysis.finetune_indobert --csv data/latih.csv --epochs 3
```

> Dataset `IndonesiaAI/offline-school-sentiment-on-indonesian-twitter` sengaja
> **tidak** didaftarkan: berbasis Twitter (menarik) tapi arti label 0/1/2-nya
> tidak terdokumentasi. Pakai `load_custom()` hanya bila kamu sudah memastikan
> sendiri maknanya — salah petakan membuat hasil terbalik.

### b. Bandingkan model (tab 🔬 Banding Model)

⚠️ **Setiap model memakai urutan label berbeda:**

| Model | id2label |
|---|---|
| `mdhugol` (bawaan proyek) | `LABEL_0/1/2` (tak informatif → pakai override) |
| `w11wo` | `0=positive, 1=neutral, 2=negative` |
| `ayame` | `0=Positive, 1=Neutral, 2=Negative` |
| `indobertweet` | `0=Negative, 1=Neutral, 2=Positive` ← **terbalik!** |

Modul `hf_models.py` **membaca `id2label` dari config saat runtime** dan menolak
jalan bila label tak bisa dipastikan — mencegah hasil terbalik tanpa disadari.

```python
from analysis.hf_datasets import load_labeled
from analysis.hf_models import compare_models, tabel_ringkas
teks, label = load_labeled("carant", limit=600, seimbang=True)
res = compare_models(teks, label, ["mdhugol", "w11wo", "indobertweet"])
```
Atau lewat tab **🔬 Banding Model** di dashboard. Tiap model diunduh ±500 MB
saat pertama dipakai.

### c. Analisis emosi (tab 😠 Emosi)

Model: `StevenLimcorn/indonesian-roberta-base-emotion-classifier` →
**marah, takut, sedih, senang, cinta**.

```powershell
python tools/setup_local_models.py --emotion   # simpan model emosi ke lokal
```
Lalu klik **Analisis emosi dokumen** di tab Emosi. Dashboard menampilkan
distribusi emosi, tren harian, heatmap emosi × sentimen, dan akun dengan emosi
marah terbanyak.

> Kenapa berguna: dua konten sama-sama "negatif" bisa sangat berbeda — yang
> memicu **kemarahan** cenderung lebih cepat viral daripada yang memicu kesedihan.

### d. Melatih di GPU gratis (Colab)

Melatih di CPU laptop lambat. Untuk data ribuan, pakai
[`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb):

1. Buka [Google Colab](https://colab.research.google.com/) → `File > Upload notebook`
2. `Runtime > Change runtime type > GPU`
3. Jalankan sel dari atas ke bawah
4. Unduh hasilnya, ekstrak ke `models/indobert-sentiment-finetuned/`
5. Set `config.yaml → sentiment.model_dir`

### e. Unggah model ke HuggingFace Hub

⚠️ **Mengunggah = mengirim ke layanan eksternal.** Repo **privat** secara bawaan.

```powershell
setx HF_TOKEN "hf_xxx"                                    # token akses Write
python tools/push_to_hub.py --repo namamu/model-ku        # PRATINJAU saja
python tools/push_to_hub.py --repo namamu/model-ku --yes  # benar-benar unggah
```
Tanpa `--yes` skrip hanya menampilkan rencana, tidak mengirim apa pun.
`--public` butuh konfirmasi ketik ulang.

> Periksa data latihmu dulu: bobot model bisa menghafal potongan data, jadi
> jangan unggah model yang dilatih dari data pribadi/sensitif.

---

## 14. Troubleshooting

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
| Word cloud / heatmap lambat | Matikan **Stemming** di sidebar, atau persempit filter platform/sentimen. |
| "Data terlalu sedikit" di tab ML | Butuh ≥20 dokumen berlabel dan ≥2 kelas. Kumpulkan data lebih dulu. |
| SMOTE "dilewati" | Kelas terkecil punya <2 sampel. Tambah data pada kelas minoritas. |
| Semua skor F1 = 1.00 | Datanya terlalu seragam/sedikit sehingga mudah ditebak — bukan hasil valid. Perbanyak & ragamkan data. |
| Stemming tidak jalan | `Sastrawi` belum terpasang: `uv pip install Sastrawi`. |
| "model lokal tak ditemukan" | Jalankan `python tools/setup_local_models.py` (sekali, ±500 MB). |
| Kata baru tak ter-stem | Tambahkan kata dasarnya ke `resources/kata_dasar_custom.txt`, lalu jalankan ulang. |
| Slang tertentu tak dikenali | Tambahkan barisnya ke `resources/slang_baku.csv` (`slang,baku`). |
| Fine-tuning kehabisan memori | Turunkan `--batch-size` (mis. 8 atau 4). |
| "Can't load the configuration of ..." | Koneksi terputus saat mengunduh model. Ulangi; model besar (±500 MB/model). |
| Tab Emosi kosong | Jalankan `python tools/setup_local_models.py --emotion`, lalu klik **Analisis emosi dokumen**. |
| Banding model: "label tidak bisa dipastikan" | Model memakai `LABEL_0/1/2` tanpa keterangan. Beri override eksplisit — **jangan** menebak urutannya. |
| Dataset HF gagal dimuat | Perlu internet. Cek juga `uv pip install datasets`. |
| Melatih sangat lambat | Wajar di CPU. Pakai notebook Colab (GPU gratis) di `notebooks/`. |

---

Butuh bantuan lebih lanjut? Lihat [README.md](README.md) untuk ringkasan arsitektur.
