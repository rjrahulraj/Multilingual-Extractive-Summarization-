import numpy as np
import pandas as pd
from typing import Dict, Any, List
import torch

# Extractive components (FAST: KMeans only)
from extractive.clustering import run_kmeans
from extractive.selector import build_summary_from_clusters

# SBERT embedding + sentence splitting (GPU)
from gensumm.gensumm_extractive import split_sentences, embed_sentences

# Ultra-speed abstractive summarizer
from abstractive.abs_summarizer import AbstractiveSummarizer

# Metrics (ROUGE / BLEU / METEOR)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# BERTScore (GPU)
from bert_score import BERTScorer

# Dataset loader
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Eval] Running on device: {DEVICE}")

# Preload BERTScore ONCE on GPU → Step 6
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=DEVICE)


# ================================================================
# ⚡ Step 3: Fast non-BERT metrics
# ================================================================
def compute_metrics_no_bert(ref: str, gen: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    s = scorer.score(ref, gen)

    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([ref.split()], gen.split(), smoothing_function=smoothie)

    try:
        meteor_v = meteor_score([ref.split()], gen.split())
    except Exception:
        meteor_v = 0.0

    return {
        "rouge1": float(s["rouge1"].fmeasure),
        "rouge2": float(s["rouge2"].fmeasure),
        "rougeL": float(s["rougeL"].fmeasure),
        "bleu": float(bleu),
        "meteor": float(meteor_v),
    }


# ================================================================
# ⭐ Step 6 — Batched BERTScore
# ================================================================
def add_bertscore_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bertscore"] = np.nan

    methods = df["method"].unique().tolist()

    for m in methods:
        mask = df["method"] == m
        gens = df.loc[mask, "summary"].tolist()
        refs = df.loc[mask, "reference"].tolist()

        if len(gens) == 0:
            continue

        print(f"🔬 Computing BERTScore for method = {m} (batch-size={len(gens)})")

        with torch.no_grad():
            _, _, F1 = bert_scorer.score(gens, refs)

        df.loc[mask, "bertscore"] = F1.cpu().numpy()

    return df


# ================================================================
# ⚡ Step 4 & 5 — Batched hybrid + abstractive generation
# ================================================================
def batch_summarize(model, texts: List[str], batch_size=4):
    outputs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        outs = model.summarize_batch(chunk, batch_size=len(chunk))
        outputs.extend(outs)
    return outputs


# ================================================================
# ⚡ Full Evaluation Pipeline (Highly Optimized)
# ================================================================
def evaluate_on_dataset(
    dataset_name: str,
    dataset_config: str = None,
    num_samples: int = 100,
    abs_model_name: str = "google/flan-t5-base",
    compression_ratio: float = 0.2,
    lambda_param: float = 0.7,
    max_cap: int = 40,
    min_sentences: int = 3,
    export_csv: str = "results.csv",
) -> Dict[str, Any]:

    print(f"\n🔍 Loading dataset: {dataset_name} ({dataset_config})")
    ds = load_dataset(dataset_name, dataset_config)
    data = ds["test"].select(range(min(num_samples, len(ds["test"]))))

    # Ultra-speed abstractive model (Step 2)
    abs_model = AbstractiveSummarizer(
        abs_model_name,
        ultra_speed=True,
        use_8bit=False,   # keep false on Windows
    )

    rows = []
    hybrid_inputs = []
    hybrid_refs = []
    abstr_inputs = []
    abstr_refs = []

    for idx, item in enumerate(data):

        print(f"\n📝 Processing sample {idx+1}/{len(data)}")

        # Step 3 — Periodic GPU cleanup
        if DEVICE == "cuda" and (idx % 10 == 0):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("🧹 Cleared GPU cache")

        doc = item["text"]
        ref = item["summary"]

        # 1) Sentence split
        sents = split_sentences(doc)
        if len(sents) < 2:
            continue

        n = len(sents)
        target = int(n * compression_ratio)
        max_sent = min(max_cap, max(min_sentences, target))

        # 2) SBERT embeddings (GPU)
        embs = embed_sentences(sents)

        # 3) Extractive (FAST: KMeans only)
        k = max(2, min(max_sent, n - 1))
        labels = run_kmeans(embs.cpu().numpy(), k)

        selected = build_summary_from_clusters(
            sents, embs.cpu().numpy(), labels, max_sent, lambda_param
        )
        ext_summary = " ".join(selected)

        rows.append({
            "method": "extractive",
            "summary": ext_summary,
            "reference": ref,
            **compute_metrics_no_bert(ref, ext_summary)
        })

        # Store hybrid generation input (Step 4)
        hybrid_inputs.append(ext_summary)
        hybrid_refs.append(ref)

        # Store abstractive full input (Step 5)
        abstr_inputs.append(doc)
        abstr_refs.append(ref)

    # -------------------------------------------------------
    # 4) Batched Hybrid Summaries
    # -------------------------------------------------------
    print("\n🚀 Generating hybrid summaries in batches…")
    hybrid_outputs = batch_summarize(abs_model, hybrid_inputs)

    for hyb, ref in zip(hybrid_outputs, hybrid_refs):
        rows.append({
            "method": "hybrid",
            "summary": hyb,
            "reference": ref,
            **compute_metrics_no_bert(ref, hyb)
        })

    # -------------------------------------------------------
    # 5) Batched Abstractive Summaries
    # -------------------------------------------------------
    print("\n🚀 Generating abstractive summaries in batches…")
    abstr_outputs = batch_summarize(abs_model, abstr_inputs)

    for summ, ref in zip(abstr_outputs, abstr_refs):
        rows.append({
            "method": "abstractive",
            "summary": summ,
            "reference": ref,
            **compute_metrics_no_bert(ref, summ)
        })

    # Convert → DataFrame
    df = pd.DataFrame(rows)

    # -------------------------------------------------------
    # ⭐ Step 6: Add BERTScore in a single batched pass
    # -------------------------------------------------------
    df = add_bertscore_column(df)

    df.to_csv(export_csv, index=False)
    print(f"\n📁 Saved results to: {export_csv}")

    agg = df.groupby("method").mean(numeric_only=True)
    print("\n📊 AVERAGE METRICS:\n", agg)

    return {"df": df, "aggregate": agg}
