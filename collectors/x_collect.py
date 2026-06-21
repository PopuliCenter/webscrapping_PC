"""Orchestrator X/Twitter: gabungkan jalur gratis (twikit) & berbayar (service).

method:
  twikit  -> hanya twikit
  service -> hanya layanan berbayar
  auto    -> coba twikit; bila kosong/gagal, fallback ke service
"""
from __future__ import annotations

import asyncio
from typing import List, Tuple

from core.models import Document
from . import x_twikit, x_service


def collect(cfg_x: dict) -> Tuple[List[Document], list]:
    if not cfg_x.get("enabled"):
        return [], []

    queries = cfg_x.get("search_queries") or []
    if not queries:
        print("  [x] tak ada search_queries di config -> dilewati")
        return [], []

    method = cfg_x.get("method", "auto").lower()
    docs: List[Document] = []
    edges: list = []

    if method in ("twikit", "auto"):
        d, e = asyncio.run(x_twikit.collect(cfg_x, queries))
        docs += d
        edges += e

    if method == "service" or (method == "auto" and not docs):
        if method == "auto":
            print("  [x] twikit nihil -> fallback ke layanan berbayar")
        d, e = x_service.collect(cfg_x, queries)
        docs += d
        edges += e

    return docs, edges
