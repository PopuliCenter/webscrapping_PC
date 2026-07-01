# Image untuk scheduler (scraping) & dashboard.
# Pakai Python 3.12 (wheel torch/lxml lebih stabil di Docker daripada 3.14).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Jakarta

WORKDIR /app

# torch versi CPU dulu (jauh lebih kecil dari default yg bawa CUDA ~2GB+)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt tzdata

COPY . .

# Default: jalankan scheduler. Service dashboard menimpa command ini di compose.
CMD ["python", "scheduler.py"]
