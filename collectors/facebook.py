"""Collector Facebook — IMPORTER dari export Meta Content Library.

PENTING — kondisi realistis:
  CrowdTangle sudah ditutup (2024). Scraping FB langsung melanggar ToS, sangat
  rapuh, dan mudah diblokir. Jalur sah untuk riset = **Meta Content Library**
  (butuh persetujuan akses peneliti/akademik via Inter-university Consortium).
  Data dari sana di-EKSPOR (CSV/JSON), lalu diimpor ke pipeline ini.

Skill ini membaca file export tersebut -> skema seragam. Tidak melakukan
scraping. Set path file di config.yaml -> facebook.import_file.
"""
from __future__ import annotations

import csv
import json
import os
from typing import List, Tuple

from core.models import Document
from .base import build_edges, extract_mentions, clean_text


def _pick(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def parse_record(rec: dict) -> Tuple[Document, list]:
    """Map satu baris export Content Library -> (Document, edges)."""
    author = str(_pick(rec, "account_name", "page_name", "author", "username"))
    text = clean_text(str(_pick(rec, "text", "message", "post_text", "content")))
    url = str(_pick(rec, "url", "post_url", "permalink"))

    doc = Document(
        platform="facebook",
        source=author,
        author=author,
        url=url,
        content=text,
        lang=str(_pick(rec, "lang", "language")),
        published_at=str(_pick(rec, "date", "created_time", "post_date")),
        engagement={
            "like": _pick(rec, "likes", "like_count", "reactions", default=0),
            "comment": _pick(rec, "comments", "comment_count", default=0),
            "share": _pick(rec, "shares", "share_count", default=0),
        },
        raw={"source": "content_library"},
    )
    edges = build_edges("facebook", author, doc.doc_id, doc.published_at,
                        mentions=extract_mentions(text))
    return doc, edges


def collect(cfg_fb: dict) -> Tuple[List[Document], list]:
    path = cfg_fb.get("import_file", "")
    if not path:
        print("  [fb] import_file kosong (butuh export Meta Content Library) -> dilewati")
        return [], []
    if not os.path.isfile(path):
        print(f"  [fb] file tidak ditemukan: {path} -> dilewati")
        return [], []

    records: List[dict] = []
    try:
        if path.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else data.get("data", [])
        else:  # CSV/TSV
            delim = "\t" if path.lower().endswith(".tsv") else ","
            with open(path, encoding="utf-8", newline="") as f:
                records = list(csv.DictReader(f, delimiter=delim))
    except Exception as e:
        print(f"  [fb] gagal baca {path}: {e}")
        return [], []

    docs: List[Document] = []
    edges: list = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        doc, e = parse_record(rec)
        docs.append(doc)
        edges.extend(e)
    print(f"  [fb] {len(docs)} post diimpor dari {os.path.basename(path)}")
    return docs, edges
