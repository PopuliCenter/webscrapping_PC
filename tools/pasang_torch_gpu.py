"""Pasang PyTorch versi GPU (CUDA) di .venv — agar fine-tuning pakai kartu grafis.

Kenapa perlu: `pip/uv install torch` dari PyPI di Windows memberi versi `+cpu`,
yang TIDAK bisa memakai GPU walaupun kartu NVIDIA ada.

Langkah otomatis:
  1. Deteksi GPU NVIDIA & versi CUDA yang didukung driver (nvidia-smi)
  2. Pilih build yang cocok (cu130 bila driver >= 13.0, cu126 bila >= 12.6)
     dengan VERSI torch yang sama seperti yang sudah terpasang
  3. Unduh wheel (~1,9 GB) dengan RESUME + cek SHA-256 (tools/unduh_model.py)
  4. Pasang ke .venv dengan uv, lalu uji: cuda aktif + hitung di GPU

Jalankan:
    python tools/pasang_torch_gpu.py            # deteksi + pasang
    python tools/pasang_torch_gpu.py --cek      # hanya periksa
    python tools/pasang_torch_gpu.py --cpu      # kembali ke versi CPU
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.unduh_model import unduh_url  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJECT_DIR, ".cache", "wheels")
VENV_PY = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
INDEX = "https://download.pytorch.org/whl/{varian}/torch/"
# (versi CUDA driver minimum, varian wheel) — dicoba dari yang terbaru
VARIAN = [((13, 0), "cu130"), ((12, 6), "cu126")]


def info_gpu() -> dict:
    """Nama GPU, VRAM, dan versi CUDA yang didukung driver."""
    if not shutil.which("nvidia-smi"):
        return {}
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if q.returncode != 0 or not q.stdout.strip():
        return {}
    nama, vram, driver = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
    kepala = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", kepala)
    cuda = tuple(int(x) for x in m.group(1).split(".")[:2]) if m else (0, 0)
    return {"nama": nama, "vram_mb": int(float(vram)), "driver": driver, "cuda": cuda}


def info_torch() -> dict:
    kode = ("import torch,json;print(json.dumps({'versi':torch.__version__,"
            "'cuda':torch.version.cuda,'aktif':torch.cuda.is_available()}))")
    r = subprocess.run([VENV_PY, "-c", kode], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def cari_uv() -> str:
    kandidat = [shutil.which("uv"),
                os.path.expandvars(r"%APPDATA%\Python\Python314\Scripts\uv.exe"),
                os.path.expandvars(r"%USERPROFILE%\.local\bin\uv.exe")]
    for k in kandidat:
        if k and os.path.isfile(k):
            return k
    raise RuntimeError("uv tidak ditemukan. Pasang dulu: pip install uv")


def cari_wheel(versi: str, varian: str) -> tuple:
    """-> (url, sha256) wheel torch untuk Python & Windows ini, atau (None, None)."""
    import requests
    from tools.unduh_model import _coba
    r = subprocess.run([VENV_PY, "-c", "import sys;print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"],
                       capture_output=True, text=True)
    py = r.stdout.strip()                    # versi Python di .venv, bukan interpreter ini
    teks = _coba(lambda: requests.get(INDEX.format(varian=varian), timeout=(10, 60)).text,
                 label=f"indeks {varian}")
    pola = rf'href="([^"]*torch-{re.escape(versi)}%2B{varian}-{py}-{py}-win_amd64\.whl)#sha256=([0-9a-f]+)"'
    m = re.search(pola, teks)
    if not m:
        return None, None
    return requests.compat.urljoin(INDEX.format(varian=varian), m.group(1)), m.group(2)


def uji_gpu() -> bool:
    kode = ("import torch,time;d=torch.device('cuda');a=torch.randn(4096,4096,device=d);"
            "torch.cuda.synchronize();t=time.time();[a@a for _ in range(20)];"
            "torch.cuda.synchronize();print(f'GPU: {torch.cuda.get_device_name(0)} | "
            "20 perkalian matriks 4096x4096: {time.time()-t:.2f} dtk')")
    r = subprocess.run([VENV_PY, "-c", kode], capture_output=True, text=True)
    print("  " + (r.stdout.strip() or r.stderr.strip()[-300:]))
    return r.returncode == 0


def main():
    gpu, tor = info_gpu(), info_torch()
    print("── Kondisi sekarang ──")
    if gpu:
        print(f"  GPU   : {gpu['nama']} | VRAM {gpu['vram_mb']} MB | driver {gpu['driver']} "
              f"| CUDA driver {'.'.join(map(str, gpu['cuda']))}")
    else:
        print("  GPU   : tidak ada GPU NVIDIA terdeteksi")
    print(f"  torch : {tor.get('versi')} | CUDA build {tor.get('cuda')} | "
          f"GPU aktif: {tor.get('aktif')}")

    if "--cek" in sys.argv:
        return
    if tor.get("aktif"):
        print("\nPyTorch sudah memakai GPU — tidak ada yang perlu dipasang.")
        uji_gpu()
        return

    uv = cari_uv()
    versi = (tor.get("versi") or "2.12.1").split("+")[0]

    if "--cpu" in sys.argv:
        subprocess.run([uv, "pip", "install", "--python", VENV_PY, "--reinstall",
                        f"torch=={versi}"], check=True)
        return
    if not gpu:
        print("\nTidak ada GPU NVIDIA — tetap memakai versi CPU.")
        return

    url = sha = varian = None
    for minimum, v in VARIAN:
        if gpu["cuda"] >= minimum:
            url, sha = cari_wheel(versi, v)
            if url:
                varian = v
                break
    if not url:
        print(f"\nTidak ada wheel torch {versi} yang cocok dengan CUDA driver "
              f"{gpu['cuda']}. Perbarui driver NVIDIA lalu coba lagi.")
        return

    nama = os.path.basename(url).replace("%2B", "+")
    print(f"\n── Mengunduh torch {versi}+{varian} (±1,9 GB, bisa dilanjutkan bila putus) ──")
    wheel = unduh_url(url, os.path.join(CACHE, nama), sha256=sha)

    print("\n── Memasang ke .venv ──")
    subprocess.run([uv, "pip", "install", "--python", VENV_PY, "--reinstall-package",
                    "torch", wheel], check=True)

    print("\n── Uji ──")
    tor = info_torch()
    print(f"  torch {tor.get('versi')} | CUDA {tor.get('cuda')} | GPU aktif: {tor.get('aktif')}")
    if tor.get("aktif") and uji_gpu():
        print("\nBerhasil. Fine-tuning kini otomatis memakai GPU.")
        print(f"Wheel disimpan di {CACHE} — boleh dihapus untuk menghemat 1,9 GB.")


if __name__ == "__main__":
    main()
