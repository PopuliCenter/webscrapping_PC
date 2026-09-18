"""Preprocessing teks Bahasa Indonesia untuk analisis sentimen & text mining.

Tahapan (urut, khas metodologi riset sentimen ID):
  1. case folding            -> huruf kecil semua
  2. cleansing               -> buang URL, mention (@user), hashtag, emoji,
                                angka, tanda baca, karakter berulang
  3. normalisasi kata baku   -> kamus slang/alay  ("gak" -> "tidak")
  4. tokenizing              -> pecah jadi token
  5. stopword removal        -> buang kata umum tak bermakna
  6. stemming (Sastrawi)     -> kembalikan ke kata dasar

Pakai:
    from analysis.preprocess import Preprocessor
    p = Preprocessor()
    p.clean("Gakk suka bangettt sama @orang ini!! 😡 http://x.co")
    p.tokens(...)        # daftar token akhir
    p.steps(...)         # hasil tiap tahap (untuk laporan/transparansi)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable

# ── Stemmer Sastrawi (opsional) ─────────────────────────────────
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _HAS_SASTRAWI = True
except Exception:
    _HAS_SASTRAWI = False


# ── Pola regex ──────────────────────────────────────────────────
RE_URL      = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION  = re.compile(r"@\w+")
RE_HASHTAG  = re.compile(r"#(\w+)")          # '#' dibuang, katanya disimpan
RE_RT       = re.compile(r"\brt\b[\s:]*", re.I)
RE_HTML     = re.compile(r"<[^>]+>|&\w+;")
RE_EMOJI    = re.compile(
    "["
    "\U0001F300-\U0001F9FF"   # simbol & piktograf
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"   # misc simbol & dingbats
    "\U0001F1E6-\U0001F1FF"   # bendera
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002B00-\U00002BFF"
    "]+", flags=re.UNICODE
)
RE_NONALPHA = re.compile(r"[^a-z\s]")        # setelah lowercase: sisakan huruf+spasi
RE_ELONG    = re.compile(r"(.)\1{2,}")       # "bangettt" -> "banget"
RE_WS       = re.compile(r"\s+")


# ── Kamus normalisasi slang/alay -> kata baku ───────────────────
SLANG_BAKU = {
    "gak": "tidak", "ga": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak",
    "enggak": "tidak", "tdk": "tidak", "tak": "tidak", "gapapa": "tidak apa apa",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "untk": "untuk",
    "krn": "karena", "karna": "karena", "krna": "karena", "sy": "saya", "aq": "saya",
    "aku": "saya", "gw": "saya", "gue": "saya", "gua": "saya", "ane": "saya",
    "lu": "kamu", "lo": "kamu", "loe": "kamu", "elo": "kamu", "kmu": "kamu",
    "km": "kamu", "situ": "kamu", "kalian": "kamu",
    "bgt": "banget", "bngt": "banget", "banget": "sangat", "bgtu": "begitu",
    "udh": "sudah", "udah": "sudah", "dah": "sudah", "blm": "belum", "blom": "belum",
    "blum": "belum", "sdh": "sudah", "tp": "tapi", "tpi": "tapi", "tapi": "tetapi",
    "jd": "jadi", "jgn": "jangan", "jgnkan": "jangankan", "klo": "kalau",
    "kalo": "kalau", "klu": "kalau", "kl": "kalau", "gmn": "bagaimana",
    "gimana": "bagaimana", "gmna": "bagaimana", "knp": "kenapa", "knapa": "kenapa",
    "kenapa": "mengapa", "emg": "memang", "emang": "memang", "mmg": "memang",
    "bnr": "benar", "bener": "benar", "bnrn": "benaran", "sm": "sama",
    "sma": "sama", "org": "orang", "orng": "orang", "hrs": "harus", "hrus": "harus",
    "bs": "bisa", "bsa": "bisa", "gabisa": "tidak bisa", "gamau": "tidak mau",
    "gatau": "tidak tahu", "gtau": "tidak tahu", "tau": "tahu", "tw": "tahu",
    "skrg": "sekarang", "skrang": "sekarang", "sekarng": "sekarang",
    "bsk": "besok", "kmrn": "kemarin", "kemaren": "kemarin", "nnti": "nanti",
    "ntar": "nanti", "bntr": "sebentar", "bentar": "sebentar",
    "dr": "dari", "dri": "dari", "ke": "ke", "kmna": "kemana", "dmn": "dimana",
    "dimna": "dimana", "sbg": "sebagai", "spt": "seperti", "kyk": "seperti",
    "kayak": "seperti", "kaya": "seperti", "kyknya": "sepertinya",
    "bkn": "bukan", "bukn": "bukan", "msh": "masih", "msih": "masih",
    "pd": "pada", "dlm": "dalam", "dlu": "dahulu", "dulu": "dahulu",
    "trs": "terus", "trus": "terus", "lg": "lagi", "lgi": "lagi",
    "sgt": "sangat", "sangatt": "sangat", "plg": "paling", "paling": "paling",
    "byk": "banyak", "bnyk": "banyak", "dikit": "sedikit", "dkit": "sedikit",
    "cm": "cuma", "cuman": "cuma", "cuma": "hanya", "aja": "saja", "aj": "saja",
    "doang": "saja", "sih": "", "deh": "", "dong": "", "nih": "ini", "tuh": "itu",
    "kok": "", "kan": "", "ya": "iya", "yaa": "iya", "yah": "iya", "iyaa": "iya",
    "mantul": "mantap", "mantab": "mantap", "keren": "bagus", "kece": "bagus",
    "jelek": "buruk", "parah": "buruk", "ancur": "hancur", "rusak": "rusak",
    "bgs": "bagus", "baguss": "bagus", "jlek": "buruk",
    "pemerintah": "pemerintah", "pemrintah": "pemerintah",
    "kk": "kakak", "gan": "", "sis": "", "bro": "", "min": "admin",
    "wkwk": "", "wkwkwk": "", "haha": "", "hahaha": "", "hehe": "", "hihi": "",
    "xixixi": "", "lol": "", "anjay": "", "anjir": "", "njir": "",
    "bgtlah": "banget", "pen": "ingin", "pengen": "ingin", "pgn": "ingin",
    "mau": "ingin", "nyari": "cari", "liat": "lihat", "ngeliat": "lihat",
    "ngga": "tidak", "engga": "tidak", "kagak": "tidak", "kga": "tidak",
}

# ── Stopword Bahasa Indonesia (inti) ────────────────────────────
STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada",
    "adalah", "akan", "atau", "juga", "sudah", "saya", "kamu", "dia", "mereka",
    "kami", "kita", "ada", "tidak", "bukan", "dalam", "oleh", "karena", "jika",
    "kalau", "agar", "supaya", "sebagai", "seperti", "bahwa", "namun", "tetapi",
    "tapi", "lalu", "kemudian", "sehingga", "maka", "bila", "saat", "ketika",
    "sambil", "hingga", "sampai", "antara", "setiap", "para", "si", "sang",
    "pun", "lah", "kah", "nya", "iya", "saja", "hanya", "cuma", "lagi", "masih",
    "sangat", "sekali", "lebih", "kurang", "paling", "banyak", "sedikit",
    "semua", "seluruh", "beberapa", "bisa", "dapat", "harus", "boleh", "mau",
    "ingin", "perlu", "sedang", "telah", "belum", "pernah", "selalu", "sering",
    "kadang", "jarang", "begitu", "begini", "demikian", "tersebut", "yaitu",
    "yakni", "adapun", "bahkan", "apalagi", "melainkan", "serta", "maupun",
    "atas", "bawah", "depan", "belakang", "luar", "sini", "situ", "sana",
    "mana", "siapa", "apa", "kenapa", "mengapa", "bagaimana", "kapan", "dimana",
    "ya", "oh", "ah", "eh", "wah", "nah", "loh", "kok", "dong", "deh", "sih",
    "nih", "tuh", "jadi", "jangan", "ayo", "mari", "tolong", "terima", "kasih",
    "orang", "hal", "cara", "hari", "waktu", "tahun", "bulan", "kali", "buah",
    "sebuah", "seorang", "suatu", "satu", "dua", "tiga", "per", "an", "nan",
}


def _load_wordlist(path: str) -> set:
    """Muat daftar kata dari file (satu kata per baris). Kosong bila gagal."""
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            return {ln.strip().lower() for ln in f if ln.strip()}
    except Exception:
        return set()


def _load_slang(path: str) -> dict:
    """Muat kamus slang dari CSV/TSV 'slang,baku' (satu pasang per baris)."""
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = re.split(r"[,\t;]", ln, maxsplit=1)
                if len(parts) == 2:
                    out[parts[0].strip().lower()] = parts[1].strip().lower()
    except Exception:
        pass
    return out


class Preprocessor:
    """Pipeline preprocessing Bahasa Indonesia yang bisa dikonfigurasi."""

    def __init__(self, remove_stopwords: bool = True, use_stemmer: bool = True,
                 normalize_slang: bool = True, min_token_len: int = 3,
                 stopword_file: str = "", slang_file: str = "",
                 extra_stopwords: Iterable[str] = ()):
        self.remove_stopwords = remove_stopwords
        self.normalize_slang = normalize_slang
        self.min_token_len = min_token_len

        self.stopwords = set(STOPWORDS_ID) | _load_wordlist(stopword_file) | set(extra_stopwords)
        self.slang = dict(SLANG_BAKU)
        self.slang.update(_load_slang(slang_file))

        self._stemmer = None
        if use_stemmer and _HAS_SASTRAWI:
            try:
                self._stemmer = StemmerFactory().create_stemmer()
            except Exception:
                self._stemmer = None
        self.stemming_active = self._stemmer is not None

    # ── Tahap-tahap ─────────────────────────────────────────────
    @staticmethod
    def case_fold(text: str) -> str:
        return (text or "").lower()

    @staticmethod
    def cleanse(text: str) -> str:
        """Buang URL, mention, hashtag(#), emoji, HTML, angka, tanda baca, huruf berulang."""
        t = text or ""
        t = RE_HTML.sub(" ", t)
        t = RE_URL.sub(" ", t)
        t = RE_RT.sub(" ", t)
        t = RE_MENTION.sub(" ", t)        # hapus mention @user
        t = RE_HASHTAG.sub(r"\1", t)      # '#ikn' -> 'ikn'
        t = RE_EMOJI.sub(" ", t)          # hapus emoji
        t = RE_NONALPHA.sub(" ", t)       # buang angka & tanda baca
        t = RE_ELONG.sub(r"\1", t)        # "bangettt" -> "banget"
        return RE_WS.sub(" ", t).strip()

    def normalize(self, tokens: list) -> list:
        """Ubah slang/alay menjadi kata baku (bisa memecah jadi beberapa kata)."""
        if not self.normalize_slang:
            return tokens
        out = []
        for t in tokens:
            if t in self.slang:
                rep = self.slang[t]
            else:
                # coba bentuk tanpa huruf ganda: "gakk"->"gak", "yaaa"->"ya"
                squeezed = re.sub(r"(.)\1+", r"\1", t)
                rep = self.slang.get(squeezed, t)
            if rep:                        # nilai "" berarti kata dibuang
                out.extend(rep.split())
        return out

    def filter_stopwords(self, tokens: list) -> list:
        if not self.remove_stopwords:
            return tokens
        return [t for t in tokens
                if t not in self.stopwords and len(t) >= self.min_token_len]

    def stem(self, tokens: list) -> list:
        if self._stemmer is None:
            return tokens
        return [_stem_cached(self._stemmer, t) for t in tokens]

    # ── API utama ───────────────────────────────────────────────
    def tokens(self, text: str) -> list:
        t = self.cleanse(self.case_fold(text))
        toks = t.split()
        toks = self.normalize(toks)
        toks = self.filter_stopwords(toks)
        toks = self.stem(toks)
        return [t for t in toks if len(t) >= self.min_token_len]

    def clean(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def steps(self, text: str) -> dict:
        """Hasil tiap tahap — berguna untuk laporan metodologi."""
        folded = self.case_fold(text)
        cleansed = self.cleanse(folded)
        tok = cleansed.split()
        normalized = self.normalize(tok)
        no_stop = self.filter_stopwords(normalized)
        stemmed = self.stem(no_stop)
        return {
            "0_asli": text,
            "1_case_folding": folded,
            "2_cleansing": cleansed,
            "3_tokenizing": tok,
            "4_normalisasi_baku": normalized,
            "5_stopword_removal": no_stop,
            "6_stemming": stemmed,
        }


@lru_cache(maxsize=50000)
def _stem_cached(stemmer, word: str) -> str:
    """Cache hasil stemming — Sastrawi lambat bila dipanggil berulang."""
    try:
        return stemmer.stem(word)
    except Exception:
        return word


# ── Deteksi near-duplicate (sinyal copy-paste / buzzer) ─────────
def content_signature(text: str, p: "Preprocessor | None" = None) -> str:
    """Tanda tangan isi: token unik terurut -> teks sama walau diacak = sama."""
    pp = p or Preprocessor(use_stemmer=False)
    toks = sorted(set(pp.tokens(text)))
    import hashlib
    return hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()
