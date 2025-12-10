# scoring.py
"""
Sentence scoring for extractive summarization.

Improved version:

Components:
- semantic_centrality: average cosine similarity to all other sentences
- textrank: graph centrality (if networkx available)
- tfidf_keyword: TF-IDF keyword salience (max TF-IDF per sentence)
- coverage: similarity to document centroid (global relevance)
- cluster_separation: cluster tightness minus cross-cluster similarity
- local_salience: similarity to cluster centroid
- novelty: reward sentences that are not near-duplicates of others

All components are min-max normalized and combined with tunable weights.
"""

from typing import List, Dict
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

# Optional libs
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_TFIDF = True
except Exception:
    TfidfVectorizer = None
    _HAS_TFIDF = False


def _normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array; returns ones if nearly constant."""
    if x.size == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-8:
        return np.ones_like(x)
    return (x - mn) / (mx - mn)


def compute_sentence_scores(
    sentences: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    weights: Dict[str, float] = None,
) -> np.ndarray:
    """
    Computes a composite importance score for each sentence.

    Components:
    - semantic_centrality: average cosine similarity to all others
    - textrank: graph centrality (if networkx available)
    - tfidf_keyword: max TF-IDF weight (keyword salience)
    - coverage: cosine similarity to document centroid
    - cluster_separation: intra-cluster tightness vs. inter-cluster similarity
    - local_salience: similarity to cluster centroid
    - novelty: 1 - max similarity to any other sentence (reward diversity)

    Args:
        sentences: list of sentence strings
        embeddings: (n, d) sentence embeddings
        labels: (n,) cluster labels
        weights: optional dictionary overriding component weights

    Returns:
        scores: (n,) final scores
    """
    n = len(sentences)
    if n == 0:
        return np.zeros(0, dtype=float)

    # -----------------------------
    # Default component weights
    # -----------------------------
    default_weights = {
        "semantic_centrality": 0.25,
        "textrank": 0.10,
        "tfidf_keyword": 0.15,
        "coverage": 0.20,
        "cluster_separation": 0.10,
        "local_salience": 0.10,
        "novelty": 0.10,
    }
    if weights is None:
        weights = default_weights
    else:
        # Merge user weights with defaults
        merged = default_weights.copy()
        merged.update(weights)
        weights = merged

    # Precompute similarity matrix
    sim = cosine_similarity(embeddings)  # (n, n)

    # -------------------------------------------------
    # 1) Semantic centrality (embedding-based)
    #     → average similarity to all sentences
    # -------------------------------------------------
    # diag is 1.0 (self-similarity); keeping it is fine as it's same for all
    semantic_centrality = sim.mean(axis=1)

    # -------------------------------------------------
    # 2) TextRank / graph centrality (optional)
    # -------------------------------------------------
    textrank = np.ones(n) / n
    if _HAS_NX:
        try:
            G = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    if sim[i, j] > 0:
                        G.add_edge(i, j, weight=float(sim[i, j]))
            pr = nx.pagerank(G, weight="weight")
            textrank = np.array([pr.get(i, 0.0) for i in range(n)], dtype=float)
        except Exception:
            # Fall back to uniform if pagerank fails
            textrank = np.ones(n) / n

    # -------------------------------------------------
    # 3) TF-IDF keyword salience (max TF-IDF per sentence)
    # -------------------------------------------------
    tfidf_keyword = np.ones(n, dtype=float)
    if _HAS_TFIDF:
        try:
            vec = TfidfVectorizer()
            tfidf = vec.fit_transform(sentences)              # (n, vocab)
            # Use max TF-IDF in each sentence to emphasize strong keywords
            tfidf_keyword = np.asarray(tfidf.max(axis=1)).reshape(-1)
        except Exception:
            pass

    # -------------------------------------------------
    # 4) Coverage / global relevance
    #     → similarity to document centroid
    # -------------------------------------------------
    doc_centroid = embeddings.mean(axis=0, keepdims=True)    # (1, d)
    coverage = cosine_similarity(embeddings, doc_centroid).reshape(-1)

    # -------------------------------------------------
    # 5) Cluster-level quality + separation + local salience
    # -------------------------------------------------
    labels_list = labels.tolist()
    unique_clusters = sorted(set(labels_list))

    cluster_separation = np.zeros(n, dtype=float)
    local_cluster_salience = np.zeros(n, dtype=float)

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        cluster_emb = embeddings[idx]
        intra = cosine_similarity(cluster_emb)  # (m, m)

        # Intra-cluster tightness
        intra_tightness = float(intra.mean())

        # Inter-cluster similarity: avg similarity to sentences not in cluster
        other_idx = np.where(labels != c)[0]
        if len(other_idx) > 0:
            inter_sim = cosine_similarity(cluster_emb, embeddings[other_idx])
            inter_mean = float(inter_sim.mean())
        else:
            inter_mean = 0.0

        # Higher is better: tight cluster with low external similarity
        sep_score = intra_tightness - inter_mean
        cluster_separation[idx] = sep_score

        # Local salience: similarity to cluster centroid
        if len(idx) > 1:
            centroid = cluster_emb.mean(axis=0, keepdims=True)
            sims_local = cosine_similarity(cluster_emb, centroid).reshape(-1)
            local_cluster_salience[idx] = sims_local

    # -------------------------------------------------
    # 6) Novelty: reward sentences that are *not* near-duplicates
    # -------------------------------------------------
    # For each sentence: max similarity to any *other* sentence
    if n > 1:
        sim_no_diag = sim.copy()
        np.fill_diagonal(sim_no_diag, -1.0)  # exclude self from max
        max_sim_others = sim_no_diag.max(axis=1)
    else:
        max_sim_others = np.zeros(n, dtype=float)

    # Novelty = 1 - normalized(max_sim_others)
    # → high novelty if max_sim_others is low
    novelty_raw = 1.0 - _normalize(max_sim_others)

    # -------------------------------------------------
    # Normalize all components to [0,1]
    # -------------------------------------------------
    components = {
        "semantic_centrality": _normalize(semantic_centrality),
        "textrank": _normalize(textrank),
        "tfidf_keyword": _normalize(tfidf_keyword),
        "coverage": _normalize(coverage),
        "cluster_separation": _normalize(cluster_separation),
        "local_salience": _normalize(local_cluster_salience),
        "novelty": _normalize(novelty_raw),
    }

    # -------------------------------------------------
    # Combine with weights
    # -------------------------------------------------
    final_scores = np.zeros(n, dtype=float)
    for name, comp in components.items():
        w = float(weights.get(name, 0.0))
        if w != 0.0:
            final_scores += w * comp

    return final_scores
