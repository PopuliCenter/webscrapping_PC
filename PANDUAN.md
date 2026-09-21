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
13. [Integrasi HuggingFace (dataset, banding model, emosi, sarkasme, intent, Colab)](#13-integrasi-huggingface)
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

### Berita panjang: dinilai per paragraf

Yang disimpan dari tiap berita bukan hanya judul, tapi juga **isi artikel penuh**
(diambil trafilatura bila `news.fetch_full_text: true`), dan penilaian memakai
judul + isi.

IndoBERT hanya bisa membaca ±512 karakter sekali jalan. Karena itu teks panjang
dipecah **per paragraf**, tiap paragraf dinilai sendiri, lalu digabung dengan
bobot panjang paragraf. Paragraf utuh sengaja tidak disatukan — kalau disatukan,
paragraf protes bisa menempel ke paragraf netral dan nadanya saling menghapus.

Hasil tambahan tersimpan di database:

| Kolom | Arti |
|---|---|
| `bagian_total` | jumlah paragraf yang dinilai |
| `bagian_negatif` / `bagian_positif` | berapa paragraf bernada negatif / positif |
| `kutipan_negatif` | paragraf paling negatif (untuk menelusuri isu) |

Ini membedakan berita yang **negatif seluruhnya** dari berita netral yang memuat
**satu kutipan keras**. Rinciannya tampil di tab **Ringkasan** dashboard.

Diuji pada 100 berita RSS nyata: 49 berita cukup panjang (rata-rata 7,5 paragraf),
**11 label berubah** — hampir semua dari netral menjadi positif/negatif. Waktunya
2,1 → 5,4 detik per 100 berita karena penilaian kini juga memakai GPU bila ada.

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

**Simpan model ke folder proyek** (sekali saja) agar jalan tanpa internet:
```powershell
python tools/setup_local_models.py --all     # semua model (±2,5 GB)
python tools/setup_local_models.py --check   # lihat mana yang sudah ada
```
Pilih sebagian dengan `--sentimen --w11wo --emotion --sarkasme --zeroshot`.
Folder `models/` sengaja di-`.gitignore` karena besar.

> **Koneksi ke HuggingFace sering terputus?** Pengunduh proyek ini
> (`tools/unduh_model.py`) mengambil satu berkas per waktu dan **melanjutkan
> dari byte terakhir** bila putus. Kalau gagal, cukup jalankan ulang perintah
> yang sama — tidak mulai dari nol. Model lain pun bisa diunduh dengan:
> `python tools/unduh_model.py <repo_id> models/<nama_folder>`

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
python tools/siapkan_data_latih.py              # buat data/latih_20k.csv & data/latih_sarkasme.csv
python -m analysis.finetune_indobert --tugas sentimen --csv data/latih_20k.csv
python -m analysis.finetune_indobert --tugas sarkasme --csv data/latih_sarkasme.csv
```
CSV berkolom `text,label` (opsional `split` = train/validation/test).
Protokolnya: urutan label **mengikuti model dasar**, **bobot kelas** untuk data
timpang, dan angka akhir dari data **uji** yang tak pernah dilihat saat latihan.

#### Melatih dengan GPU laptop (NVIDIA)

`pip`/`uv install torch` di Windows memberi PyTorch versi **`+cpu`**, yang tidak
bisa memakai GPU walau kartu NVIDIA ada. Pasang versi CUDA sekali:
```powershell
python tools/pasang_torch_gpu.py --cek    # lihat GPU & versi torch sekarang
python tools/pasang_torch_gpu.py          # deteksi driver, unduh (±1,9 GB, bisa resume), pasang, uji
```
Setelah itu fine-tuning otomatis memakai GPU (`Perangkat : cuda`).

**Alur yang disarankan — latih sebagai KANDIDAT, bandingkan, baru pakai:**
```powershell
python -m analysis.finetune_indobert --tugas sarkasme --csv data/latih_sarkasme.csv --epochs 8 --output models/sarkasme-kandidat
python -m analysis.sarcasm --bandingkan models/sarkasme-kandidat

python -m analysis.finetune_indobert --tugas sentimen --csv data/latih_20k.csv --epochs 3 --output models/sentimen-kandidat
python -m analysis.hf_models --bandingkan models/sentimen-kandidat
```
Pembanding memakai **uji McNemar** pada teks yang sama: selisih F1 kecil sering
hanya kebetulan. Kandidat dipakai hanya bila lebih baik secara signifikan.

**VRAM kecil (mis. RTX 3050 4 GB):** batch 16 + `max_len 128` memakai ±2,3–2,9 GB.
Kalau muncul pesan VRAM habis, kecilkan batch dan tambah akumulasi supaya batch
efektif tetap sama: `--batch-size 8 --akumulasi 2`.

**Hasil nyata di RTX 3050 Laptop (sarkasme, 1.878 tweet, 8 epoch):** ±6 menit,
VRAM puncak 2,25 GB. F1 sarkas 0,7589 vs patokan 0,7273 — tapi McNemar
p = 0,708 (**tidak signifikan**) dan salah cap tweet tulus justru naik (41 vs 34).
Jadi patokan tetap dipakai. Pelajarannya: data latih umum tidak cukup — yang
dibutuhkan contoh berlabel **dari topikmu sendiri**.

**Menambah data sarkasme dari dataset publik:**
```powershell
python tools/siapkan_data_latih.py sarkasme-plus reddit            # -> data/latih_sarkasme_reddit.csv
python tools/siapkan_data_latih.py sarkasme-plus reddit argilla    # + 9,7k tweet politik 2025
python -m analysis.finetune_indobert --tugas sarkasme --csv data/latih_sarkasme_reddit.csv --epochs 4 --output models/sarkasme-kandidat-reddit
python -m analysis.sarcasm --bandingkan models/sarkasme-kandidat-reddit
```
Sumber (semua HuggingFace): `reddit` = 14k komentar r/indonesia berlabel tag `/s`
(Apache-2.0), `argilla` = tweet 2025 berlabel manual (labelnya berisik),
`sintetis` = 250 tweet sarkas buatan LLM. Data tambahan **hanya masuk data latih**;
validasi & uji tetap tweet resmi, dan teks yang sama dengan data uji dibuang
(cegah bocor). Pembanding kini juga menilai di uji Reddit (2.824 komentar).

Hasil nyata (4 epoch, ±20–30 menit per model di RTX 3050):

| Model | F1 uji Twitter | salah cap tulus | F1 uji Reddit |
|---|---|---|---|
| patokan (w11wo) | 0,727 | 34 | 0,369 |
| + Reddit (13k latih) | 0,696 (p = 0,66, setara) | 31 | **0,606** (p ≈ 1e-21) |
| + Reddit + Argilla (22k) | 0,692 (p = 0,46, setara) | 36 | 0,593 |

Artinya: data tambahan **tidak** membuat model lebih jago di tweet (setara), tapi
**jauh lebih tahan di teks non-Twitter** (komentar forum — mirip komentar IG/FB).
Data Argilla tidak menambah apa pun (labelnya berisik). Patokan tetap dipakai
untuk X; bila fokusmu komentar IG/FB, `models/sarkasme-kandidat-reddit` layak
dipakai — ganti nama foldernya jadi `models/sarkasme-finetuned`.
Dataset GitHub sengaja tidak dipakai: umumnya tanpa lisensi, kecil, atau labelnya
otomatis/berisik.

> **Tips kecepatan:** memuat model & pustaka CUDA dari HDD terasa lambat di awal
> (bisa beberapa menit). Bila laptopmu punya SSD dan HDD, letakkan proyek (atau
> minimal folder `models/` dan `.venv/`) di SSD.
>
> **Laptop (Optimus):** saat mulai, GPU "bangun" dari mode hemat daya — `nvidia-smi`
> kadang gagal sesaat. Wajar. Colokkan charger saat melatih; di baterai GPU dibatasi.
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

Melatih di CPU laptop lambat. [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb)
melatih **sentimen** atau **sarkasme** di GPU gratis — pilih lewat `TUGAS`:

| `TUGAS` | Data | Model dasar | Hasil ke folder |
|---|---|---|---|
| `"sentimen"` | 21.000 berlabel manusia (seimbang) | `w11wo` (pemenang uji banding) | `models/indobert-sentiment-finetuned/` |
| `"sarkasme"` | 2.684 tweet, split resmi | IndoBERTweet | `models/sarkasme-finetuned/` |

1. Buka [Google Colab](https://colab.research.google.com/) → `File > Upload notebook`
2. `Runtime > Change runtime type > GPU`
3. Set `TUGAS`, lalu `Runtime > Run all` — data diunduh otomatis
   (atau `SUMBER_DATA = "upload"` untuk CSV dari `data/`)
4. Unduh zip hasilnya, ekstrak ke folder di tabel
5. Sentimen: set `config.yaml → sentiment.model_dir`. Sarkasme: otomatis dipakai.

**Pembanding sarkasme:** model jadi `w11wo` mencapai **F1 0,7273** pada 538
tweet uji resmi (sudah diverifikasi ulang, cocok persis dengan angka resmi).
Hasil latihmu layak dipakai bila menyamai atau melampauinya.

Notebook dibuat oleh `tools/buat_notebook_colab.py` — ubah skrip itu, bukan
notebook-nya langsung, lalu jalankan ulang.

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

### f. Sarkasme, intent & intensitas 5 tingkat (tab 🎯 Intent & Sarkasme)

Lapisan di atas sentimen biasa. Siapkan modelnya sekali:
```powershell
python tools/setup_local_models.py --sarkasme --zeroshot --w11wo
```
Lalu klik tombol di tab **🎯 Intent & Sarkasme**, atau lewat terminal:
```powershell
python -m analysis.sarcasm              # coba pada contoh
python -m analysis.sarcasm --ambang     # pilih ambang (validasi) & laporkan (uji)
python -m analysis.zeroshot             # coba intent pada contoh
python -m analysis.zeroshot --kalibrasi # kalibrasi ulang ambang 5 tingkat
```

**Seberapa bisa dipercaya — semua diukur, bukan diklaim:**

| Fitur | Cara | Hasil terukur | Catatan |
|---|---|---|---|
| **Sarkasme** | model jadi `w11wo` (IndoBERT, dilatih dari tweet) | F1 **0,727**, presisi 0,74 — 538 tweet uji resmi | ±1 dari 4 tanda "sarkas" keliru |
| **Intensitas 5 tingkat** | selisih logit `w11wo`, ambang dikalibrasi | tepat **65,5%**, meleset ≤1 tingkat **88,7%**, Spearman 0,82 — 600 ulasan uji PRDECT-ID | tingkat tengah (2–4) paling sulit |
| **Intent** | zero-shot mDeBERTa, 7 kategori | **belum bisa diukur** | tak ada data berlabel yang relevan |

**Hal yang perlu kamu tahu:**

- **Label model sarkasme dibuktikan, bukan ditebak.** Config-nya hanya
  `LABEL_0/1`. Diuji dua kemungkinan pada data uji: `LABEL_1 = sarkas` → F1 0,727;
  kebalikannya → F1 0,140. Hasilnya cocok persis dengan angka resmi pembuatnya.
- **Sarkasme cenderung salah mencap keluhan tulus.** Contoh: *"jalan di depan
  rumah rusak, mohon segera diperbaiki"* terbaca sarkas (0,888). Menaikkan ambang
  hampir tak membantu — dari 0,5 ke 0,99 presisi hanya naik 0,74 → 0,81, dan tak
  ada ambang yang mencapai 0,85. Kesalahannya "yakin", jadi obatnya **melatih
  ulang dengan contoh keluhan vs sarkasme dari datamu** (notebook Colab,
  `TUGAS = "sarkasme"`). Ambang aktif 0,9 (dipilih di data validasi) tersimpan di
  `resources/ambang_sarkasme.json`.
- Kolom **"Sarkas tapi dilabeli POSITIF"** di dashboard adalah daftar paling
  berguna: pujian yang sebenarnya sindiran — sentimennya kemungkinan terbalik.
- **Intent = indikasi, bukan kepastian.** Pada 7 contoh jelas yang ditulis
  manual, 5 sesuai; skor keyakinan umumnya rendah (0,37–0,57). "Kritik" dan
  "keluhan" sering tertukar. Zero-shot juga **lambat di CPU** (tiap teks diuji
  terhadap 7 hipotesis) — batasi jumlah dokumen atau jalankan di GPU.
- **Kenapa intensitas tidak memakai zero-shot?** Sudah diuji pada data yang sama:
  zero-shot hanya **17,2%** tepat (di bawah tebak acak 20%) walau arahnya benar
  (Spearman 0,82 — sama dengan polaritas). Ia menghindari label "sangat …" dan
  memilih tingkat tengah. Polaritas terkalibrasi: **65,5%**.
- **Intensitas diukur dengan rating bintang ulasan produk** sebagai pendekatan.
  Ulasan produk ≠ opini politik; ambangnya bisa dikalibrasi ulang bila kamu punya
  data berlabel 5 tingkat dari topikmu sendiri.

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
| Melatih sangat lambat | Cek `python tools/pasang_torch_gpu.py --cek` — kalau torch `+cpu`, pasang versi GPU. Tanpa GPU: notebook Colab. |
| "VRAM GPU habis" | `--batch-size 8 --akumulasi 2` (batch efektif sama), atau `--max-len 96`. |
| `nvidia-smi` gagal sesaat | GPU laptop sedang bangun dari mode hemat daya — tunggu beberapa detik. |
| Latihan lama sebelum mulai | Pustaka CUDA & model dimuat dari HDD. Pindahkan proyek ke SSD. |

---

Butuh bantuan lebih lanjut? Lihat [README.md](README.md) untuk ringkasan arsitektur.
