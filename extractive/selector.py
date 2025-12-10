# selector.py (IMPROVED VERSION)
"""
Advanced sentence selection for extractive summarization.

Includes:
- GPU cosine similarity (when embeddings are torch tensors)
- Cluster-aware centrality + positional weighting
- Softmax-based cluster quota
- Improved MMR for global selection
- Better candidate pooling for coverage + non-redundancy
"""

from typing import List
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------
# GPU cosine similarity (fallback to CPU)
# -------------------------------------------------------
def fast_cosine(emb):
    if isinstance(emb, torch.Tensor):
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        sim = emb @ emb.T
        return sim.detach().cpu().numpy()
    else:
        return cosine_similarity(emb)


# -------------------------------------------------------
# Sentence Scoring (centrality + cluster + position)
# -------------------------------------------------------
def compute_sentence_scores(sentences, embeddings, labels):
    n = len(sentences)
    sim = fast_cosine(embeddings)

    # Centrality
    centrality = sim.sum(axis=1)

    # Cluster importance
    cluster_sizes = np.bincount(labels)
    cluster_weight = cluster_sizes / cluster_sizes.sum()
    cluster_weight = cluster_weight[labels]

    # Positional bias (lead-bias for news datasets)
    position = np.linspace(1.0, 0.5, n)

    # Final combined score
    score = (
        0.55 * centrality +
        0.30 * cluster_weight +
        0.15 * position
    )

    return score


# -------------------------------------------------------
# Enhanced MMR
# -------------------------------------------------------
def mmr_selection(embeddings, candidate_indices, k, lambda_param=0.6):
    if isinstance(embeddings, torch.Tensor):
        emb = embeddings.detach().cpu().numpy()
    else:
        emb = embeddings

    sim = fast_cosine(emb)
    cand_sim = sim[np.ix_(candidate_indices, candidate_indices)]

    selected = []
    remaining = list(range(len(candidate_indices)))

    # Start with the highest-centrality candidate
    centrality = cand_sim.sum(axis=1)
    first = int(np.argmax(centrality))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < k and remaining:
        mmr_scores = []
        for r in remaining:
            rel = centrality[r]
            div = max(cand_sim[r][s] for s in selected)
            mmr = lambda_param * rel - (1 - lambda_param) * div
            mmr_scores.append((mmr, r))

        best = max(mmr_scores)[1]
        selected.append(best)
        remaining.remove(best)

    # Map back to original sentence indices
    final_indices = [candidate_indices[i] for i in selected]
    return sorted(final_indices)


# -------------------------------------------------------
# Main cluster-aware selector
# -------------------------------------------------------
def build_summary_from_clusters(
    sentences: List[str],
    embeddings,
    labels,
    max_sent: int,
    lambda_param: float = 0.7,
    cluster_weights=None,
) -> List[str]:

    n = len(sentences)
    if n == 0:
        return []

    # Compute global sentence score
    scores = compute_sentence_scores(sentences, embeddings, labels)

    clusters = np.unique(labels)

    # -----------------------------------------
    # SOFTMAX cluster quota allocation
    # -----------------------------------------
    cluster_sizes = np.array([np.sum(labels == c) for c in clusters])
    exp_weights = np.exp(cluster_sizes / 10.0)
    quota_fraction = exp_weights / exp_weights.sum()

    quotas = {
        c: max(1, int(round(quota_fraction[i] * max_sent)))
        for i, c in enumerate(clusters)
    }

    # Adjust quotas to match exactly max_sent
    while sum(quotas.values()) > max_sent:
        largest = max(quotas, key=lambda c: quotas[c])
        if quotas[largest] > 1:
            quotas[largest] -= 1
        else:
            break

    while sum(quotas.values()) < max_sent:
        smallest = min(quotas, key=lambda c: quotas[c])
        quotas[smallest] += 1

    # -----------------------------------------
    # Candidate pool per cluster
    # -----------------------------------------
    candidate_indices = []

    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        # Rank sentences inside cluster
        cluster_scores = scores[idx]
        order = idx[np.argsort(-cluster_scores)]

        # Add 2× quota to candidate pool
        num_cand = min(len(order), quotas[c] * 2)
        candidate_indices.extend(order[:num_cand])

    candidate_indices = sorted(set(candidate_indices))

    # Safety fallback
    if len(candidate_indices) == 0:
        top_k = np.argsort(-scores)[:max_sent]
        return [sentences[i] for i in sorted(top_k)]

    # -----------------------------------------
    # Global diversity via MMR
    # -----------------------------------------
    selected_indices = mmr_selection(
        embeddings, candidate_indices, max_sent, lambda_param=lambda_param
    )

    return [sentences[i] for i in selected_indices]
