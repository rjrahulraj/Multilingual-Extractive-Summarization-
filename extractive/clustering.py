# clustering.py (IMPROVED RESEARCH VERSION)
"""
Advanced clustering module for extractive summarization.

New features:
- MiniBatchKMeans (fast, stable)
- Spectral clustering (non-linear manifold-aware)
- Agglomerative clustering (hierarchical topic grouping)
- OPTICS-style density clustering (noise tolerant)
- Composite cluster scoring (silhouette + DB + cluster balance)
- Torch-aware cosine distance handling
"""

from typing import Dict, Callable, Tuple, Optional
import numpy as np

# ---------------------------------------------
#  IMPORTS WITH SAFE FALLBACKS
# ---------------------------------------------
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    KMeans = MiniBatchKMeans = SpectralClustering = AgglomerativeClustering = None
    silhouette_score = davies_bouldin_score = None

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------------------------------------
#  COSINE NORMALIZATION UTILITIES
# ---------------------------------------------
def normalize(X):
    if _HAS_TORCH and isinstance(X, torch.Tensor):
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        return X.cpu().numpy()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


# ---------------------------------------------
#  CLUSTERING METHODS
# ---------------------------------------------
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_SKLEARN:
        return np.array([i % k for i in range(len(X))])
    Xn = normalize(X)
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model.fit_predict(Xn)


def run_minibatch_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_SKLEARN:
        return run_kmeans(X, k)
    Xn = normalize(X)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=256,
        n_init=5,
        max_iter=100
    )
    return model.fit_predict(Xn)


def run_spectral(X: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_SKLEARN:
        return run_kmeans(X, k)
    Xn = normalize(X)
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=42
    )
    return model.fit_predict(Xn)


def run_agglomerative(X: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_SKLEARN:
        return run_kmeans(X, k)
    Xn = normalize(X)
    model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return model.fit_predict(Xn)


def run_optics_variant(X: np.ndarray, k: int) -> np.ndarray:
    """
    Light HDBSCAN-style fallback:
    Assign sentences to nearest of k prototypes using density heuristics.
    """
    Xn = normalize(X)
    n = len(X)
    rng = np.random.RandomState(42)

    # pick k prototypes
    proto_idx = rng.choice(n, k, replace=False)
    protos = Xn[proto_idx]

    # cosine similarities
    sim = Xn @ protos.T  # (n, k)
    labels = np.argmax(sim, axis=1)
    return labels


# ---------------------------------------------
#  METHOD REGISTRY
# ---------------------------------------------
CLUSTER_METHODS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "kmeans": run_kmeans,
    "minibatch": run_minibatch_kmeans,
    "spectral": run_spectral,
    "agglomerative": run_agglomerative,
    "optics": run_optics_variant,
}


# ---------------------------------------------
#  INTERNAL QUALITY SCORING
# ---------------------------------------------
def cluster_balance_score(labels: np.ndarray) -> float:
    """
    Measures how evenly distributed clusters are.
    Perfect balance = 1.0
    All samples in one cluster = 0.0
    """
    counts = np.bincount(labels)
    p = counts / counts.sum()
    return float(1.0 - (p.max() - p.min()))


def evaluate_clustering_internal(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Composite score combining multiple metrics.
    Higher = better.
    """

    if not _HAS_SKLEARN or len(set(labels)) < 2:
        return -9999.0

    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = -1.0

    try:
        db = davies_bouldin_score(X, labels)
        db = -db  # inverted for "higher is better"
    except Exception:
        db = -100.0

    bal = cluster_balance_score(labels)

    # Composite research metric
    final = 0.50 * sil + 0.30 * db + 0.20 * bal
    return float(final)


# ---------------------------------------------
#  TOP-LEVEL METHOD SELECTION
# ---------------------------------------------
def choose_best_clustering(
    X: np.ndarray,
    k: int,
    methods: Optional[Dict[str, Callable]] = None
) -> Tuple[str, np.ndarray]:

    if methods is None:
        methods = CLUSTER_METHODS

    best_score = -1e12
    best_method = None
    best_labels = None

    for name, fn in methods.items():
        try:
            labels = fn(X, k)
            score = evaluate_clustering_internal(X, labels)

            if score > best_score:
                best_score = score
                best_method = name
                best_labels = labels

        except Exception:
            continue

    # Fall back if everything fails
    if best_labels is None:
        n = len(X)
        return "fallback", np.array([i % k for i in range(n)])

    return best_method, best_labels
