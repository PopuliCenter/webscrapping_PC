"""Kirim laporan lewat email (SMTP). MATI secara bawaan.

Mengirim email adalah tindakan keluar — sekali terkirim tidak bisa ditarik.
Karena itu ada tiga kunci yang semuanya harus dibuka sendiri olehmu:

  1. `laporan.email.enabled: true` di config.yaml
  2. berkas `secrets/email.yaml` berisi kredensial SMTP
  3. daftar `penerima` tidak kosong

Kalau salah satu belum terpenuhi, fungsi ini menolak dengan pesan jelas dan
TIDAK mengirim apa pun.

Gmail: pakai App Password (bukan kata sandi akun), aktifkan 2FA lebih dulu.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMPIRAN_MAKS = 15 * 1024 * 1024        # batas lazim lampiran email


class EmailTidakSiap(Exception):
    """Prasyarat pengiriman belum lengkap — sengaja bukan error diam-diam."""


def _kredensial(path: str = "") -> dict:
    import yaml
    path = path or os.path.join(PROJECT_DIR, "secrets", "email.yaml")
    if not os.path.isfile(path):
        raise EmailTidakSiap(
            f"{os.path.relpath(path, PROJECT_DIR)} belum ada. Salin dari "
            f"secrets/email.yaml.example lalu isi.")
    data = yaml.safe_load(open(path, encoding="utf-8")) or {}
    for wajib in ("host", "port", "user", "password"):
        if not data.get(wajib):
            raise EmailTidakSiap(f"'{wajib}' kosong di secrets/email.yaml")
    return data


def kirim(folder: str, cfg_email: dict, judul: str = "", ringkasan: str = "",
          uji_saja: bool = False) -> dict:
    """Kirim isi folder laporan ke penerima. -> ringkasan tindakan.

    `uji_saja=True` memeriksa semua prasyarat dan menyusun pesannya, tapi
    BERHENTI sebelum mengirim — dipakai untuk memastikan setelan benar tanpa
    mengganggu siapa pun.
    """
    if not cfg_email.get("enabled"):
        raise EmailTidakSiap("laporan.email.enabled masih false di config.yaml")
    penerima = [p for p in (cfg_email.get("penerima") or []) if str(p).strip()]
    if not penerima:
        raise EmailTidakSiap("daftar laporan.email.penerima masih kosong")
    kred = _kredensial()

    pesan = EmailMessage()
    pesan["Subject"] = judul or f"Laporan monitoring — {os.path.basename(folder)}"
    pesan["From"] = kred.get("dari") or kred["user"]
    pesan["To"] = ", ".join(penerima)
    pesan.set_content(ringkasan or "Laporan monitoring terlampir.")

    dilampirkan, dilewati = [], []
    for nama in sorted(os.listdir(folder)):
        jalur = os.path.join(folder, nama)
        if not os.path.isfile(jalur):
            continue
        if os.path.getsize(jalur) > LAMPIRAN_MAKS:
            dilewati.append(nama)
            continue
        with open(jalur, "rb") as f:
            pesan.add_attachment(f.read(), maintype="application",
                                 subtype="octet-stream", filename=nama)
        dilampirkan.append(nama)

    if uji_saja:
        return {"terkirim": False, "uji_saja": True, "penerima": penerima,
                "lampiran": dilampirkan, "dilewati": dilewati}

    konteks = ssl.create_default_context()
    port = int(kred["port"])
    if port == 465:
        with smtplib.SMTP_SSL(kred["host"], port, context=konteks) as s:
            s.login(kred["user"], kred["password"])
            s.send_message(pesan)
    else:
        with smtplib.SMTP(kred["host"], port) as s:
            s.starttls(context=konteks)
            s.login(kred["user"], kred["password"])
            s.send_message(pesan)
    return {"terkirim": True, "penerima": penerima, "lampiran": dilampirkan,
            "dilewati": dilewati}
