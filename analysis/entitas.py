"""Ekstraksi tokoh/lembaga dan KUTIPAN dari berita — "siapa bilang apa".

Kebutuhan sehari-hari media specialist: bukan cuma "isu ini negatif", tapi
"siapa yang bicara, dan kalimat mana yang dikutip". Modul ini mengambil dua hal:

  1. Kutipan langsung  : kalimat dalam tanda kutip + siapa pengucapnya.
  2. Tokoh & lembaga   : nama berawalan huruf besar, disaring dengan gelar
                         jabatan dan kata umum.

Caranya berbasis ATURAN, bukan model NER. Alasannya jujur: pola kutipan bahasa
Indonesia sangat teratur ("...," kata X / ujar X / menurut X), jadi aturan
sederhana sudah akurat dan bisa diperiksa manusia — tanpa unduhan model 400 MB
dan tanpa kotak hitam. Konsekuensinya: nama yang tidak lazim, atau kalimat tanpa
tanda kutip, bisa terlewat. Angka ketepatannya diukur di tools/periksa_fitur.py.
"""
from __future__ import annotations

import re
from collections import Counter

# Kata kerja penanda kutipan dalam berita Indonesia.
KATA_UJAR = (r"kata|ujar|tutur|jelas|ungkap|tegas|ucap|sebut|imbuh|tambah|"
             r"papar|terang|lanjut|pungkas|beber|kritik|klaim|bantah|akui")

# "..." kata Prabowo   /   "..." ujar Menteri Keuangan Purbaya
_KUTIP_SESUDAH = re.compile(
    r'[""""]([^""""]{15,400})[""""]\s*,?\s*(?:' + KATA_UJAR + r')\w*\s+'
    # gelar huruf kecil boleh menyela: "kata juru bicara Kementerian ..., Nama"
    r'(?:[a-z]{2,12}\s+){0,3}'
    r'((?:[A-Z][\w.\'-]+[,\s]{0,2}){1,7})')
# Menurut Prabowo, "..."   /   Prabowo mengatakan, "..."
_KUTIP_SEBELUM = re.compile(
    r'(?:Menurut|menurut)\s+((?:[A-Z][\w.\'-]+\s?){1,5}),?\s*[""""]([^""""]{15,400})[""""]')

GELAR = ("presiden", "wakil presiden", "wapres", "menteri", "menko", "menlu",
         "menkeu", "gubernur", "wali kota", "walikota", "bupati", "kapolri",
         "kapolda", "panglima", "ketua", "direktur", "dirut", "juru bicara",
         "jubir", "anggota", "komisioner", "sekjen", "kepala", "dubes",
         "pengamat", "ekonom", "analis", "peneliti", "pakar",
         # pangkat & jabatan lapis kedua ("Kapolri Jenderal X", "Menteri Keuangan X")
         "jenderal", "komjen", "irjen", "brigjen", "laksamana", "marsekal",
         "keuangan", "luar negeri", "dalam negeri", "pertahanan", "perdagangan",
         "perindustrian", "pertanian", "investasi", "bumn", "kesehatan",
         "pendidikan", "koordinator", "utama", "jenderal polisi")

# Kata berhuruf besar yang bukan nama: awal kalimat, hari, bulan, dsb.
BUKAN_NAMA = {
    "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
    "januari", "februari", "maret", "april", "mei", "juni", "juli",
    "agustus", "september", "oktober", "november", "desember",
    "namun", "sementara", "selain", "sebelumnya", "menurut", "dalam", "pada",
    "dengan", "untuk", "dari", "ini", "itu", "kata", "ujar", "hal", "saat",
    "ketika", "setelah", "sebagai", "adapun", "sebab", "karena", "bahkan",
    "meski", "walau", "jika", "kalau", "bila", "agar", "supaya", "sedangkan",
    "tak", "tidak", "bukan", "akan", "sudah", "telah", "masih", "juga",
    "berita", "foto", "video", "baca", "lihat", "simak", "advertisement",
    # jabatan/kata peran yang sering tertangkap sebagai nama
    "head", "chief", "senior", "chairman", "director", "manager", "analyst",
    "research", "economist", "sementara", "adapun", "terkait", "hingga",
}


def _kupas_gelar(n: str) -> str:
    """Buang gelar di depan, berlapis: "Menteri Keuangan Purbaya" -> "Purbaya"."""
    for _ in range(4):
        rendah = n.lower()
        for g in sorted(GELAR, key=len, reverse=True):
            if rendah.startswith(g + " "):
                n = n[len(g):].strip()
                break
        else:
            break
    return n


def _bersih_nama(nama: str) -> str:
    """Rapikan nama pembicara dari tangkapan regex.

    Tiga masalah nyata yang ditangani, semuanya ditemukan pada berita sungguhan:
      - kalimat berikutnya ikut tertangkap  -> "Bahlil. Karena"
      - nama orang ada SETELAH koma         -> "Gubernur Banten, Andra Soni"
      - gelar berlapis                      -> "Kapolri Jenderal Listyo ..."
    """
    teks = " ".join((nama or "").split())
    # potong di akhir kalimat (titik di ujung kata, bukan inisial "H.")
    potong = []
    for k in teks.split():
        potong.append(k)
        inti = k.strip(" .,:;\"'")
        if k.endswith(".") and len(inti) > 2:
            break
    teks = " ".join(potong)

    # pecah per koma: "Gubernur Banten, Andra Soni, usai acara"
    ruas = []
    for r in teks.split(","):
        r = _kupas_gelar(r.strip(" .,:;\"'"))
        kata = [k for k in r.split() if k]
        while kata and kata[-1].lower().strip(".") in BUKAN_NAMA:
            kata.pop()
        while kata and kata[0].lower().strip(".") in BUKAN_NAMA:
            kata.pop(0)
        r = " ".join(kata)
        # hanya ruas yang benar-benar tampak seperti nama (berawalan kapital)
        if r and r[0].isupper():
            ruas.append(r)
    if not ruas:
        return ""
    # bila ada beberapa ruas, nama orang biasanya yang TERAKHIR
    return ruas[-1] if len(ruas) > 1 else ruas[0]


def kutipan(teks: str, maks: int = 20) -> list:
    """-> [{'pembicara', 'kutipan'}] dari kutipan langsung di dalam teks."""
    if not teks:
        return []
    hasil, terlihat = [], set()
    for isi, nama in _KUTIP_SESUDAH.findall(teks):
        n = _bersih_nama(nama)
        if len(n) < 3 or n.lower() in BUKAN_NAMA:
            continue
        kunci = isi[:60]
        if kunci in terlihat:
            continue
        terlihat.add(kunci)
        hasil.append({"pembicara": n, "kutipan": " ".join(isi.split())})
    for nama, isi in _KUTIP_SEBELUM.findall(teks):
        n = _bersih_nama(nama)
        if len(n) < 3 or n.lower() in BUKAN_NAMA:
            continue
        kunci = isi[:60]
        if kunci not in terlihat:
            terlihat.add(kunci)
            hasil.append({"pembicara": n, "kutipan": " ".join(isi.split())})
    return hasil[:maks]


_NAMA = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def entitas(teks: str, min_kata: int = 2) -> list:
    """Nama tokoh/lembaga (rangkaian kata berhuruf besar). -> [(nama, jumlah)].

    `min_kata=2` menuntut minimal dua kata, karena satu kata berhuruf besar
    sering hanya awal kalimat. Nama satu kata yang penting (mis. "Prabowo")
    tetap tertangkap lewat pasangan seperti "Presiden Prabowo".
    """
    if not teks:
        return []
    c = Counter()
    for calon in _NAMA.findall(teks):
        kata = calon.split()
        if len(kata) < min_kata:
            continue
        if kata[0].lower() in BUKAN_NAMA:
            kata = kata[1:]
            if len(kata) < min_kata:
                continue
        nama = _bersih_nama(" ".join(kata))
        if len(nama.split()) < min_kata or nama.lower() in BUKAN_NAMA:
            continue
        c[nama] += 1
    return c.most_common()


def dari_dokumen(df, kolom: str = "teks", maks_kutipan: int = 200) -> tuple:
    """-> (daftar_kutipan, tabel_entitas) untuk satu DataFrame dokumen."""
    kutip, ent = [], Counter()
    for _, r in df.iterrows():
        teks = str(r.get(kolom) or "")
        for k in kutipan(teks):
            if len(kutip) < maks_kutipan:
                k = dict(k)
                k["source"] = r.get("source", "")
                k["judul"] = str(r.get("title") or "")[:90]
                k["sentimen"] = r.get("sentiment_label", "")
                kutip.append(k)
        for nama, n in entitas(teks):
            ent[nama] += n
    return kutip, ent.most_common(40)
