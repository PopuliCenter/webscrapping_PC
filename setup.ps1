# Setup lokal (Windows PowerShell) memakai uv:
# Python 3.12 MANDIRI (lepas dari Python sistem) + semua dependency, di folder proyek.
# Jalankan sekali:  .\setup.ps1
#
# Kenapa uv, bukan `python -m venv`? venv meminjam Python sistem, jadi rusak bila
# Python sistem di-upgrade/dihapus. uv mengunduh Python-nya sendiri -> tahan.

$ErrorActionPreference = "Stop"

# Pasang uv bila belum ada
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Memasang uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

Write-Host "==> Mengunduh Python 3.12 mandiri (dikelola uv)..."
uv python install 3.12

Write-Host "==> Membuat .venv (Python 3.12 mandiri)..."
uv venv --python 3.12

Write-Host "==> Meng-install requirements.txt (torch/IndoBERT berat, sabar)..."
uv pip install -r requirements.txt

Write-Host ""
Write-Host "Selesai. Untuk memakai:"
Write-Host "  .\.venv\Scripts\Activate.ps1     # aktifkan"
Write-Host "  python run_once.py               # uji (Berita langsung jalan)"
Write-Host "  streamlit run dashboard/app.py   # dashboard"
