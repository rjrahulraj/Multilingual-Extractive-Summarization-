import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os

# -----------------------------
# Optional imports / global flags
# -----------------------------
try:
    import pdfplumber
except Exception:
    pdfplumber = None

SBERT_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _HAS_SBERT = True
except Exception:
    _HAS_SBERT = False

try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    _HAS_OCR = True
except Exception:
    pytesseract = None
    Image = None
    _HAS_OCR = False

try:
    from langdetect import detect as _lang_detect
    def detect_language(text: str) -> str:
        try:
            return _lang_detect(text)
        except Exception:
            return "unknown"
except Exception:
    def detect_language(text: str) -> str:
        return "unknown"

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False

from sklearn.metrics.pairwise import cosine_similarity

# clustering + metrics
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    _HAS_SKLEARN_CLUSTER = True
except Exception:
    KMeans = None
    silhouette_score = None
    davies_bouldin_score = None
    _HAS_SKLEARN_CLUSTER = False

# fuzzy c-means
try:
    from fcmeans import FCM
    _HAS_FCM = True
except Exception:
    FCM = None
    _HAS_FCM = False

# TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_TFIDF = True
except Exception:
    TfidfVectorizer = None
    _HAS_TFIDF = False


# ============================================================
#           TEXT EXTRACTION & SENTENCE SPLITTING
# ============================================================

def extract_text_pdf(path: str, do_ocr: bool = False) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for PDF parsing.")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            if do_ocr and (not page_text.strip()) and _HAS_OCR:
                try:
                    pil = page.to_image(resolution=200).original
                    ocr_text = pytesseract.image_to_string(pil)
                    text += ocr_text + "\n"
                except Exception:
                    pass
    return text


def extract_text_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(lines)


def extract_text_image(path: str) -> str:
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def extract_text(path: str, do_ocr: bool = False) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(path, do_ocr=do_ocr)
    elif ext == ".docx":
        return extract_text_docx(path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_image(path)
    else:
        raise ValueError("Unsupported file type")


def split_sentences(text: str, min_words: int = 3) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) < 4:
        sents = re.split(r"[.!?;\n]+", text)
    out = []
    for s in sents:
        s = s.strip()
        if s and len(s.split()) >= min_words:
            out.append(s)
    return out


def clean_heading(h: str) -> str:
    h = h.strip().replace(":", "")
    return h if len(h) >= 2 else "General"


def detect_sections(file_path: Optional[str], full_text: str) -> List[Tuple[str, str]]:
    # simple heuristic: split by blank lines, first line is heading
    parts = re.split(r"\n\s*\n+", full_text.strip())
    sections = []
    for p in parts:
        if len(p.strip()) < 50:
            continue
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        head = clean_heading(lines[0]) if lines else "General"
        body = "\n".join(lines[1:]) if len(lines) > 1 else p
        sections.append((head, body))
    return sections or [("General", full_text.strip())]


def embed_sentences(sentences: List[str], prefer_sbert: bool = True) -> np.ndarray:
    if prefer_sbert and _HAS_SBERT and SBERT_MODEL:
        return SBERT_MODEL.encode(sentences, show_progress_bar=False)

    dim = 384
    arr = []
    for s in sentences:
        rnd = np.random.RandomState(abs(hash(s)) % (2**31))
        arr.append(rnd.normal(size=(dim,)))
    return np.vstack(arr)


# ============================================================
#           ADVANCED CLUSTERING + INTERNAL METRICS
# ============================================================

def run_kmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_SKLEARN_CLUSTER:
        return np.array([i % k for i in range(len(embeddings))])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    return km.fit_predict(embeddings)


def run_fuzzy_cmeans(embeddings: np.ndarray, k: int) -> np.ndarray:
    if not _HAS_FCM:
        return run_kmeans(embeddings, k)
    fcm = FCM(n_clusters=k, random_state=42)
    fcm.fit(embeddings)
    return fcm.predict(embeddings)


def run_harmonic_kmeans(embeddings: np.ndarray, k: int, max_iter=50) -> np.ndarray:
    n = len(embeddings)
    k = max(1, min(k, n - 1))
    rng = np.random.RandomState(42)
    centroids = embeddings[rng.choice(n, k, replace=False)]
    for _ in range(max_iter):
        dist = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)
        weights = 1 / (dist + 1e-8)
        labels = np.argmax(weights, axis=1)
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                centroids[c] = embeddings[rng.randint(0, n)]
            else:
                w = weights[idx, c][:, None]
                centroids[c] = (embeddings[idx] * w).sum(axis=0) / (w.sum() + 1e-8)
    return labels


def run_rough_kmeans(embeddings: np.ndarray, k: int, threshold=0.7, max_iter=30) -> np.ndarray:
    n = len(embeddings)
    k = max(1, min(k, n - 1))
    rng = np.random.RandomState(123)
    centroids = embeddings[rng.choice(n, k, replace=False)]
    for _ in range(max_iter):
        sim = cosine_similarity(embeddings, centroids)
        labels = np.argmax(sim, axis=1)
        for c in range(k):
            idx_upper = np.where(sim[:, c] >= threshold)[0]
            if len(idx_upper):
                centroids[c] = embeddings[idx_upper].mean(axis=0)
            else:
                idx = np.where(labels == c)[0]
                if len(idx):
                    centroids[c] = embeddings[idx].mean(axis=0)
                else:
                    centroids[c] = embeddings[rng.randint(0, n)]
    return labels


_CLUSTER_METHODS = {
    "kmeans": run_kmeans,
    "fuzzy": run_fuzzy_cmeans,
    "harmonic": run_harmonic_kmeans,
    "rough": run_rough_kmeans,
}


def evaluate_clustering_internal(embeddings, labels):
    if not _HAS_SKLEARN_CLUSTER or len(set(labels)) < 2:
        return -9999, 9999
    try:
        sil = float(silhouette_score(embeddings, labels))
        db = float(davies_bouldin_score(embeddings, labels))
        return sil, db
    except Exception:
        return -9999, 9999


def choose_best_clustering(embeddings: np.ndarray, k: int):
    best = None
    best_score = -1e9
    for name, fn in _CLUSTER_METHODS.items():
        try:
            labels = fn(embeddings, k)
            sil, db = evaluate_clustering_internal(embeddings, labels)
            score = sil - db
            if score > best_score:
                best_score = score
                best = (name, labels)
        except Exception:
            pass
    if best is None:
        n = len(embeddings)
        return "fallback", np.array([i % k for i in range(n)])
    return best


# ============================================================
#           ADVANCED SENTENCE SCORING (A+B+C)
# ============================================================

def compute_sentence_scores(sentences, embeddings, labels):
    """
    Stronger clustering-aware scoring combining:
    - TextRank (global graph centrality)
    - TF-IDF weighting
    - Global semantic relevance (to document centroid)
    - Local cluster salience (importance within its cluster)
    - Cluster tightness (quality of the cluster)
    - Anti-redundancy term (penalize highly average sentences)
    """

    n = len(sentences)
    if n == 0:
        return np.zeros(0)

    sim = cosine_similarity(embeddings)

    # -----------------------------
    # 1) TEXT RANK
    # -----------------------------
    textrank = np.ones(n) / n
    if _HAS_NX:
        try:
            G = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    if sim[i, j] > 0:
                        G.add_edge(i, j, weight=sim[i, j])
            pr = nx.pagerank(G, weight="weight")
            textrank = np.array([pr.get(i, 0.0) for i in range(n)])
        except Exception:
            pass

    # -----------------------------
    # 2) TF-IDF
    # -----------------------------
    tfidf_scores = np.ones(n)
    if _HAS_TFIDF:
        try:
            vec = TfidfVectorizer()
            tfidf = vec.fit_transform(sentences)
            tfidf_scores = np.asarray(tfidf.mean(axis=1)).reshape(-1)
        except Exception:
            pass

    # -----------------------------
    # 3) GLOBAL SEMANTIC RELEVANCE
    # -----------------------------
    doc_centroid = embeddings.mean(axis=0, keepdims=True)
    global_relevance = cosine_similarity(embeddings, doc_centroid).reshape(-1)

    # -----------------------------
    # 4) CLUSTER QUALITY SCORE
    # -----------------------------
    cluster_quality = np.zeros(n)
    unique_clusters = sorted(set(labels.tolist()))
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        cluster_emb = embeddings[idx]
        intra = cosine_similarity(cluster_emb)
        tightness = intra.mean()  # higher → more coherent cluster
        cluster_quality[idx] = tightness

    # -----------------------------
    # 5) LOCAL CLUSTER SALIENCE
    # -----------------------------
    local_cluster_salience = np.zeros(n)
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        cluster_emb = embeddings[idx]
        cluster_centroid = cluster_emb.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_emb, cluster_centroid).reshape(-1)
        local_cluster_salience[idx] = sims

    # -----------------------------
    # 6) ANTI-REDUNDANCY (NOVELTY)
    # -----------------------------
    redundancy_penalty = np.array([
        -np.mean(sim[i]) for i in range(n)
    ])

    # -------------- Normalize helper --------------
    def norm(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn < 1e-8:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    components = {
        "textrank": norm(textrank),
        "tfidf": norm(tfidf_scores),
        "global_rel": norm(global_relevance),
        "cluster_quality": norm(cluster_quality),
        "local_salience": norm(local_cluster_salience),
        "anti_redundancy": norm(redundancy_penalty),
    }

    # -------------------------------
    # Final weighted scoring
    # -------------------------------
    final_scores = (
        0.25 * components["textrank"] +
        0.20 * components["tfidf"] +
        0.20 * components["global_rel"] +
        0.15 * components["local_salience"] +
        0.10 * components["cluster_quality"] +
        0.10 * components["anti_redundancy"]
    )

    return final_scores



# ============================================================
#           MMR + DIVERSITY-BASED SELECTION
# ============================================================

def mmr_selection(embeddings, candidate_indices, k, lambda_param=0.7):
    if not candidate_indices:
        return []
    k = min(k, len(candidate_indices))

    cand_embs = embeddings[candidate_indices]
    sim = cosine_similarity(cand_embs, cand_embs)

    selected_local = []
    remaining_local = list(range(len(candidate_indices)))

    while len(selected_local) < k and remaining_local:
        mmr_scores = []
        for idx in remaining_local:
            if not selected_local:
                diversity = 0.0
            else:
                diversity = max(sim[idx][s] for s in selected_local)
            relevance = sim[idx].sum()
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr, idx))
        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected_local.append(chosen)
        remaining_local.remove(chosen)

    return sorted(candidate_indices[i] for i in selected_local)


def build_summary_from_clusters(
    sentences,
    embeddings,
    labels,
    max_sent,
    lambda_param
):
    """
    Cluster-aware + diversity-based selection:

    1. Compute advanced sentence scores (cluster-aware).
    2. Allocate a quota of sentences per cluster proportional to its size.
    3. From each cluster, take top (quota * 2) sentences as candidates.
    4. Run global MMR on the union of candidates to pick final max_sent.
    """
    n = len(sentences)
    if n == 0:
        return []

    scores = compute_sentence_scores(sentences, embeddings, labels)
    unique_clusters = sorted(set(labels.tolist()))

    # --------------------------
    # 1) Cluster-wise quota
    # --------------------------
    cluster_sizes = {c: int(np.sum(labels == c)) for c in unique_clusters}
    total = float(sum(cluster_sizes.values()))

    # initial proportional quotas
    quotas = {}
    for c in unique_clusters:
        prop = cluster_sizes[c] / total
        quotas[c] = max(1, int(round(prop * max_sent)))

    # adjust to match exactly max_sent
    current_total = sum(quotas.values())
    # if too many, reduce from largest clusters
    while current_total > max_sent:
        # choose cluster with largest quota > 1
        candidate_cs = [c for c in unique_clusters if quotas[c] > 1]
        if not candidate_cs:
            break
        c_max = max(candidate_cs, key=lambda c: quotas[c])
        quotas[c_max] -= 1
        current_total -= 1
    # if too few, increase small clusters
    while current_total < max_sent:
        c_min = min(unique_clusters, key=lambda c: quotas[c])
        quotas[c_min] += 1
        current_total += 1

    # --------------------------
    # 2) Build candidate pool
    # --------------------------
    candidate_indices = []
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        cluster_scores = scores[idx]
        # sort indices by descending score
        order = idx[np.argsort(-cluster_scores)]
        # take up to 2×quota from this cluster as candidates
        num_cand = min(len(order), quotas[c] * 2)
        candidate_indices.extend(order[:num_cand])

    candidate_indices = sorted(set(candidate_indices))
    if not candidate_indices:
        # fallback: just take top scores globally
        order = np.argsort(-scores)[:max_sent]
        return [sentences[i] for i in sorted(order)]

    # --------------------------
    # 3) Global MMR over candidates
    # --------------------------
    selected_indices = mmr_selection(
        embeddings,
        candidate_indices,
        k=max_sent,
        lambda_param=lambda_param,
    )

    return [sentences[i] for i in selected_indices]



# ============================================================
#                EXTRACTIVE SUMMARY API
# ============================================================

def _fallback_centrality_summary(sentences, embeddings, max_sent):
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
    lambda_param: float = 0.7
):
    n = len(sentences)

    if n == 0:
        return "", "none"
    if n <= max_sent:
        return " ".join(sentences), "no-clustering"

    # adaptive λ for better diversity
    if abs(lambda_param - 0.7) < 1e-6:
        if n < 10:
            lambda_param = 0.8
        elif n < 20:
            lambda_param = 0.65
        else:
            lambda_param = 0.55

    k = max(2, min(max_sent, n - 1))

    try:
        method, labels = choose_best_clustering(embeddings, k)
        sentences_out = build_summary_from_clusters(
            sentences, embeddings, labels, max_sent, lambda_param
        )
        return " ".join(sentences_out), method
    except Exception:
        return _fallback_centrality_summary(sentences, embeddings, max_sent), "fallback"


# ============================================================
#                DOCUMENT-LEVEL EXTRACTIVE WRAPPER
# ============================================================

def summarize_document(
    file_path: str,
    prefer_sbert: bool = True,
    do_ocr: bool = False,
    compression_ratio: float = 0.35,
    max_cap: int = 40,
    min_sentences: int = 8,
    lambda_param: float = 0.7,
):
    full_text = extract_text(file_path, do_ocr=do_ocr)
    sections = detect_sections(file_path, full_text)
    out = []

    for head, content in sections:
        sents = split_sentences(content)
        if not sents:
            continue

        n = len(sents)
        target = int(round(n * compression_ratio))
        max_sent = max(min_sentences, min(max_cap, target))
        max_sent = min(max_sent, n)

        embeddings = embed_sentences(sents, prefer_sbert=prefer_sbert)
        summary, method = extractive_summary(
            sents, embeddings, max_sent, lambda_param=lambda_param
        )
        out.append(f"### {clean_heading(head)}\n{summary}\n")

    return "\n".join(out)


# ============================================================
#     CLUSTERING COMPARISON: METRICS + PLOTS (FULL DOCUMENT)
# ============================================================

# ---- Metric libraries (optional) ----
try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    rouge_scorer = None
    _HAS_ROUGE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    _HAS_NLTK = True
except Exception:
    nltk = None
    sentence_bleu = None
    SmoothingFunction = None
    meteor_score = None
    _HAS_NLTK = False

try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    bertscore_score = None
    _HAS_BERTSCORE = False

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    plt = None
    _HAS_PLT = False


def _compute_rouge(reference: str, candidate: str) -> Dict[str, Optional[float]]:
    if not _HAS_ROUGE:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge1": float(scores["rouge1"].fmeasure),
            "rouge2": float(scores["rouge2"].fmeasure),
            "rougeL": float(scores["rougeL"].fmeasure),
        }
    except Exception:
        return {"rouge1": None, "rouge2": None, "rougeL": None}


def _compute_bleu(reference: str, candidate: str) -> Optional[float]:
    if not _HAS_NLTK or sentence_bleu is None:
        return None
    try:
        smoothie = SmoothingFunction().method4 if SmoothingFunction else None
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie))
    except Exception:
        return None


def _compute_meteor(reference: str, candidate: str) -> Optional[float]:
    if not _HAS_NLTK or meteor_score is None:
        return None
    try:
        return float(meteor_score([reference.split()], candidate.split()))
    except Exception:
        return None


def _compute_bertscore(reference: str, candidate: str, lang: str = "en") -> Optional[float]:
    if not _HAS_BERTSCORE or bertscore_score is None:
        return None
    try:
        P, R, F1 = bertscore_score([candidate], [reference], lang=lang, rescale_with_baseline=True)
        return float(F1.mean().item())
    except Exception:
        return None


def compare_clusterings_for_document(
    file_path: str,
    reference_summary: str,
    lang: str = "en",
    prefer_sbert: bool = True,
    do_ocr: bool = False,
    compression_ratio: float = 0.35,
    max_cap: int = 40,
    min_sentences: int = 8,
    lambda_param: float = 0.7,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs each clustering algorithm (kmeans, fuzzy, harmonic, rough)
    for extractive summarization over the FULL document (no sections),
    and compares them via BLEU, ROUGE, METEOR, BERTScore.

    Returns:
        {
          "kmeans": {"summary": ..., "bleu": ..., "rouge1": ..., ...},
          "fuzzy":  {...},
          "harmonic": {...},
          "rough": {...},
        }
    """
    full_text = extract_text(file_path, do_ocr=do_ocr)
    sentences = split_sentences(full_text)
    results: Dict[str, Dict[str, Any]] = {}

    if not sentences:
        return results

    n = len(sentences)
    target = int(round(n * compression_ratio))
    max_sent = max(min_sentences, min(max_cap, target))
    max_sent = min(max_sent, n)

    embeddings = embed_sentences(sentences, prefer_sbert=prefer_sbert)

    k = max(2, min(max_sent, n - 1))

    ref = (reference_summary or "").strip()
    if not ref:
        raise ValueError("reference_summary must be a non-empty gold summary string.")

    for name, fn in _CLUSTER_METHODS.items():
        try:
            labels = fn(embeddings, k)
            selected_sents = build_summary_from_clusters(
                sentences, embeddings, labels, max_sent, lambda_param
            )
            summary_text = " ".join(selected_sents).strip()

            rouge = _compute_rouge(ref, summary_text)
            bleu = _compute_bleu(ref, summary_text)
            meteor = _compute_meteor(ref, summary_text)
            bscore = _compute_bertscore(ref, summary_text, lang=lang)

            results[name] = {
                "summary": summary_text,
                "bleu": bleu,
                "meteor": meteor,
                "bertscore": bscore,
                **rouge,
            }
        except Exception as e:
            results[name] = {
                "summary": "",
                "error": str(e),
                "bleu": None,
                "meteor": None,
                "bertscore": None,
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
            }

    return results


def plot_clustering_comparison(
    clustering_results: Dict[str, Dict[str, Any]],
    title: str = "Clustering Method Comparison (Extractive Summaries)"
):
    """
    Given the output of compare_clusterings_for_document(...),
    plots a bar chart for BLEU, METEOR, BERTScore, ROUGE-1/2/L
    across all clustering methods.
    """
    if not _HAS_PLT or plt is None:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    methods = list(clustering_results.keys())
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore"]

    # Prepare data
    values = {m: [] for m in metrics}
    for method in methods:
        data = clustering_results[method]
        for metric in metrics:
            v = data.get(metric, None)
            values[metric].append(0.0 if v is None else v)

    x = np.arange(len(methods))
    width = 0.12

    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[metric], width, label=metric.upper())

    plt.xticks(x + width * 2.5, methods, fontsize=11)
    plt.ylabel("Score", fontsize=13)
    plt.xlabel("Clustering Method", fontsize=13)
    plt.title(title, fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
