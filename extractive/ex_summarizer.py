# ex_summarizer.py
"""
High-level extractive summarization API (sentence-level).

This module assumes:
- You already have a list of sentences.
- You already have sentence embeddings (from SBERT, LaBSE, etc.).

It:
- Chooses the best clustering method (kmeans, fuzzy, harmonic, rough).
- Uses cluster-aware scoring and MMR-based selection.
- Provides a clean `extractive_summary` function.
"""

from typing import List, Tuple, Optional
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False

from .clustering import choose_best_clustering
from .selector import build_summary_from_clusters


def _fallback_centrality_summary(
    sentences: List[str],
    embeddings: np.ndarray,
    max_sent: int,
) -> str:
    """
    Fallback summary using simple graph centrality (TextRank-like)
    or sum of similarities if networkx is not available.
    """
    n = len(sentences)
    if n == 0:
        return ""
    if n <= max_sent:
        return " ".join(sentences)

    sim = cosine_similarity(embeddings)

    if _HAS_NX:
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(sim[i, j]))
        pr = nx.pagerank(G, weight="weight")
        scores = np.array([pr.get(i, 0.0) for i in range(n)])
    else:
        scores = sim.sum(axis=1)

    idx = np.argsort(-scores)[:max_sent]
    return " ".join(sentences[i] for i in sorted(idx))


def extractive_summary(
    sentences: List[str],
    embeddings: np.ndarray,
    max_sent: int,
    lambda_param: float = 0.7,
    force_k: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Main extractive summarization function.

    Args:
        sentences: list of sentence strings
        embeddings: (n, d) sentence embeddings
        max_sent: desired number of sentences in summary
        lambda_param: MMR diversity parameter
        force_k: optionally force a specific number of clusters k

    Returns:
        summary_text: final extractive summary as a single string
        method_name: name of clustering method used (or 'fallback')
    """
    n = len(sentences)

    if n == 0:
        return "", "none"
    if n <= max_sent:
        return " ".join(sentences), "no-clustering"

    # Adaptive λ for better diversity depending on length
    if abs(lambda_param - 0.7) < 1e-6:
        if n < 10:
            lambda_param = 0.8
        elif n < 20:
            lambda_param = 0.65
        else:
            lambda_param = 0.55

    # Number of clusters
    if force_k is not None:
        k = max(2, min(force_k, n - 1))
    else:
        k = max(2, min(max_sent, n - 1))

    try:
        method, labels = choose_best_clustering(embeddings, k)
        selected_sentences = build_summary_from_clusters(
            sentences,
            embeddings,
            labels,
            max_sent=max_sent,
            lambda_param=lambda_param,
        )
        return " ".join(selected_sentences), method
    except Exception:
        # Safety fallback to centrality-based summary
        return _fallback_centrality_summary(sentences, embeddings, max_sent), "fallback"
