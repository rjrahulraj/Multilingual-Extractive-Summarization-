"""
evaluation_summarization.py

Compare Extractive vs Hybrid vs Abstractive summarization
using BLEU, ROUGE, METEOR, BERTScore and plots.
"""

from typing import Dict, Any, List
import numpy as np

# Core summarization utilities
from gensumm.gensumm_extractive import (
    extract_text,
    split_sentences,
    embed_sentences,
    extractive_summary,
)

# Abstractive summarizer
from gensumm.abstractive import generate_abstractive_summary

# ----------------- Metric libraries (optional) -----------------

# ROUGE
try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except Exception:
    rouge_scorer = None
    _HAS_ROUGE = False

# BLEU + METEOR
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

# BERTScore
try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    bertscore_score = None
    _HAS_BERTSCORE = False

# Plotting
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    plt = None
    _HAS_PLT = False


# ============================================================
#                     METRIC HELPERS
# ============================================================

def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
    if not _HAS_ROUGE:
        return {"rouge1": None, "rouge2": None, "rougeL": None}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def compute_bleu(reference: str, candidate: str) -> float:
    if not _HAS_NLTK:
        return None

    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    smoothie = SmoothingFunction().method4 if SmoothingFunction else None

    try:
        return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie))
    except Exception:
        return None


def compute_meteor(reference: str, candidate: str) -> float:
    if not _HAS_NLTK:
        return None

    try:
        return float(meteor_score([reference.split()], candidate.split()))
    except Exception:
        return None


def compute_bertscore(reference: str, candidate: str, lang: str = "en") -> float:
    if not _HAS_BERTSCORE:
        return None
    try:
        P, R, F1 = bertscore_score([candidate], [reference], lang=lang, rescale_with_baseline=True)
        return float(F1.mean().item())
    except Exception:
        return None


def compute_all_metrics(reference: str, candidate: str, lang: str = "en") -> Dict[str, Any]:
    reference = (reference or "").strip()
    candidate = (candidate or "").strip()

    if not reference or not candidate:
        return {m: None for m in ["bleu", "meteor", "bertscore", "rouge1", "rouge2", "rougeL"]}

    rouge_scores = compute_rouge(reference, candidate)

    return {
        "bleu": compute_bleu(reference, candidate),
        "meteor": compute_meteor(reference, candidate),
        "bertscore": compute_bertscore(reference, candidate, lang),
        **rouge_scores,
    }


# ============================================================
#             SUMMARY GENERATION FOR EACH METHOD
# ============================================================

def _generate_extractive_summary_for_doc(
    full_text: str,
    compression_ratio: float = 0.20,
    max_cap: int = 40,
    min_sentences: int = 3,
    lambda_param: float = 0.7,
) -> str:
    sents = split_sentences(full_text)
    if not sents:
        return ""

    n = len(sents)
    target = int(round(n * compression_ratio))
    max_sent = max(min_sentences, min(max_cap, target))

    embs = embed_sentences(sents, prefer_sbert=True)
    summary_text, _ = extractive_summary(
        sents, embs, max_sent=max_sent, lambda_param=lambda_param
    )
    return summary_text


def _generate_hybrid_summary_for_doc(
    full_text: str,
    compression_ratio: float = 0.20,
    max_cap: int = 40,
    min_sentences: int = 3,
    lambda_param: float = 0.7,
    abs_model_name: str = "facebook/bart-large-cnn",
) -> str:

    extractive = _generate_extractive_summary_for_doc(
        full_text, compression_ratio, max_cap, min_sentences, lambda_param
    )
    if not extractive.strip():
        return ""

    return generate_abstractive_summary(extractive, model_name=abs_model_name)


def _generate_abstractive_only_summary_for_doc(
    full_text: str,
    abs_model_name: str = "facebook/bart-large-cnn",
) -> str:
    return generate_abstractive_summary(full_text, model_name=abs_model_name)


# ============================================================
#            MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_document_summaries(
    file_path: str,
    reference_summary: str,
    lang: str = "en",
    compression_ratio: float = 0.20,
    max_cap: int = 40,
    min_sentences: int = 3,
    lambda_param: float = 0.7,
    abs_model_name: str = "facebook/bart-large-cnn",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates Extractive, Hybrid, and Abstractive summarization.
    Returns ROUGE/BLEU/METEOR/BERTScore + actual summaries.
    """

    # ------------------------------------------------------------
    # 1. Read document only once
    # ------------------------------------------------------------
    full_text = extract_text(file_path, do_ocr=False)
    if not full_text.strip():
        return {
            "extractive": {},
            "hybrid": {},
            "abstractive": {},
            "_summaries": {
                "extractive": "",
                "hybrid": "",
                "abstractive": ""
            }
        }

    # ------------------------------------------------------------
    # 2. Generate Extractive Summary
    # ------------------------------------------------------------
    extractive = _generate_extractive_summary_for_doc(
        full_text,
        compression_ratio=compression_ratio,
        max_cap=max_cap,
        min_sentences=min_sentences,
        lambda_param=lambda_param,
    )

    # ------------------------------------------------------------
    # 3. Hybrid = Extractive → Abstractive
    # ------------------------------------------------------------
    hybrid = _generate_hybrid_summary_for_doc(
        full_text,
        compression_ratio=compression_ratio,
        max_cap=max_cap,
        min_sentences=min_sentences,
        lambda_param=lambda_param,
        abs_model_name=abs_model_name,
    )

    # ------------------------------------------------------------
    # 4. Fully Abstractive Summary (direct)
    # ------------------------------------------------------------
    abstractive = _generate_abstractive_only_summary_for_doc(
        full_text,
        abs_model_name=abs_model_name,
    )

    # ------------------------------------------------------------
    # 5. Compute all evaluation metrics
    # ------------------------------------------------------------
    metrics = {
        "extractive": compute_all_metrics(reference_summary, extractive, lang=lang),
        "hybrid": compute_all_metrics(reference_summary, hybrid, lang=lang),
        "abstractive": compute_all_metrics(reference_summary, abstractive, lang=lang),
        "_summaries": {
            "extractive": extractive,
            "hybrid": hybrid,
            "abstractive": abstractive,
        },
    }

    return metrics



# ============================================================
#                     CLEAN FINAL PLOTTING
# ============================================================

def plot_method_comparison(results: dict, title: str = "Extractive vs Hybrid vs Abstractive — Metric Comparison"):
    if not _HAS_PLT:
        print("matplotlib not installed.")
        return

    methods = ["extractive", "hybrid", "abstractive"]
    metrics = ["rouge1", "rouge2", "rougeL", "bleu", "meteor", "bertscore"]

    # Prepare data
    scores = {metric: [] for metric in metrics}
    for method in methods:
        metrics_dict = results.get(method, {})
        for metric in metrics:
            scores[metric].append(metrics_dict.get(metric, 0.0))

    # Plot
    plt.figure(figsize=(14, 10))
    x = np.arange(len(methods))
    width = 0.12

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, scores[metric], width, label=metric.upper())

    plt.xticks(x + width * 2.5, methods, fontsize=12)
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Summarization Method", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
