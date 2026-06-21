"""Social Network Analysis dari tabel `interactions`.

Bangun graf terarah (siapa -> siapa) lalu hitung metrik kunci ala Drone Emprit:
  - Top aktor by degree / betweenness (siapa paling berpengaruh/jadi jembatan)
  - Deteksi komunitas (Louvain) -> klaster/cluster opini
  - Ringkasan graf (node, edge, density)

Dipakai dashboard (Fase 3) & bisa dijalankan mandiri:
    python -m analysis.sna
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import networkx as nx

try:
    import community as community_louvain   # python-louvain
    _HAS_LOUVAIN = True
except Exception:
    _HAS_LOUVAIN = False


def build_graph(db_path: str, platform: Optional[str] = None,
                edge_types: Optional[list] = None) -> nx.DiGraph:
    """Bangun DiGraph berbobot dari interactions. Bobot = jumlah interaksi."""
    conn = sqlite3.connect(db_path)
    q = "SELECT src_actor, dst_actor, edge_type FROM interactions WHERE 1=1"
    params = []
    if platform:
        q += " AND platform=?"; params.append(platform)
    if edge_types:
        q += f" AND edge_type IN ({','.join('?' for _ in edge_types)})"
        params += edge_types
    rows = conn.execute(q, params).fetchall()
    conn.close()

    G = nx.DiGraph()
    for src, dst, etype in rows:
        if not src or not dst:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1, edge_type=etype)
    return G


def top_actors(G: nx.DiGraph, metric: str = "degree", n: int = 15) -> list:
    """Aktor teratas. metric: degree | in_degree | betweenness | pagerank."""
    if G.number_of_nodes() == 0:
        return []
    if metric == "in_degree":
        scores = dict(G.in_degree(weight="weight"))
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G, weight="weight")
    elif metric == "pagerank":
        scores = nx.pagerank(G, weight="weight")
    else:
        scores = dict(G.degree(weight="weight"))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


def detect_communities(G: nx.DiGraph) -> dict:
    """Partisi komunitas (Louvain) pada versi tak-berarah. -> {node: community_id}."""
    if G.number_of_nodes() == 0:
        return {}
    UG = G.to_undirected()
    if _HAS_LOUVAIN:
        return community_louvain.best_partition(UG, weight="weight")
    # Fallback: connected components bila louvain tak terpasang
    return {n: i for i, comp in enumerate(nx.connected_components(UG)) for n in comp}


def summary(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    parts = detect_communities(G)
    return {
        "nodes": n,
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 5) if n > 1 else 0,
        "communities": len(set(parts.values())) if parts else 0,
    }


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    G = build_graph(cfg["storage"]["db_path"], platform="twitter")
    print("Ringkasan:", summary(G))
    print("\nTop aktor (degree):")
    for a, s in top_actors(G, "degree"):
        print(f"  {a}: {s}")
    print("\nTop aktor (betweenness/jembatan):")
    for a, s in top_actors(G, "betweenness"):
        print(f"  {a}: {round(s, 4)}")
