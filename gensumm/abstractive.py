"""
abstractive.py
- Abstractive summarization module
- Supports multiple HF models (BART, PEGASUS, T5 / FLAN-T5, etc.)
- Handles long texts via chunking + summary-of-summaries
"""

from typing import List, Dict, Any
import math

try:
    import torch
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False
    torch = None
    pipeline = None

# Cache pipelines by model name
_SUMMARIZERS: Dict[str, Any] = {}


# ------------------------------------------------------------------
#                    DEVICE & PIPELINE HELPERS
# ------------------------------------------------------------------

def _get_device() -> int:
    """
    Returns device index for transformers pipeline:
    - 0 for first GPU if available
    - -1 for CPU
    """
    if torch is not None and torch.cuda.is_available():
        return 0
    return -1


def _is_t5_like(model_name: str) -> bool:
    """
    Rough heuristic: T5 / FLAN-T5 models use text-to-text format
    and usually want a 'summarize:' prefix.
    """
    lower = model_name.lower()
    return ("t5" in lower) or ("flan" in lower)


def _get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Returns a cached HF summarization pipeline for the given model name.
    Works for:
      - facebook/bart-large-cnn
      - google/pegasus-xsum, google/pegasus-cnn_dailymail
      - google/flan-t5-base, google/flan-t5-large, etc.
      - long models like google/long-t5, google/bigbird-pegasus-...
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "transformers is required for abstractive summarization. "
            "Install with: pip install transformers sentencepiece"
        )

    if model_name not in _SUMMARIZERS:
        device = _get_device()
        _SUMMARIZERS[model_name] = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
    return _SUMMARIZERS[model_name]


# ------------------------------------------------------------------
#                    CHUNKING & LENGTH LOGIC
# ------------------------------------------------------------------

def _estimate_max_chars_for_model(model_name: str) -> int:
    """
    Very rough heuristic: how many characters per chunk to feed the model.

    We don't know exact tokenization here, so we approximate:
    - BART / PEGASUS / T5 typical max_length ≈ 512–1024 tokens
    - We assume ~4 chars per token → 2000–4000 characters per chunk.
    """
    name = model_name.lower()
    if "bigbird" in name or "longt5" in name or "long-t5" in name:
        # Long sequence models
        return 6000
    elif "pegasus" in name:
        return 3000
    elif "t5" in name or "flan" in name:
        return 2500
    else:
        # default for BART etc.
        return 2000


def _chunk_text(text: str, model_name: str) -> List[str]:
    """
    Simple character-based chunking for long documents.
    Keeps sentence boundaries where possible.
    """
    text = (text or "").strip()
    if not text:
        return []

    max_chars = _estimate_max_chars_for_model(model_name)
    if len(text) <= max_chars:
        return [text]

    # sentence-level splitting to avoid cutting in the middle
    sentences = text.split(". ")
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        # add back the dot we split on
        segment = sent + ". "
        if current and len(current) + len(segment) > max_chars:
            chunks.append(current.strip())
            current = segment
        else:
            current += segment

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _adaptive_lengths(
    summarizer,
    input_text: str,
    max_summary_tokens: int,
    min_summary_tokens: int,
) -> (int, int):
    """
    Adapt max_length / min_length based on the input length in tokens
    to avoid HF warnings like:
    'max_length is set to 256, but input_length is only 68...'
    """
    try:
        tokenizer = summarizer.tokenizer
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        input_len = len(input_ids)
    except Exception:
        # If anything fails, just return provided lengths
        return max_summary_tokens, min_summary_tokens

    # We typically want summary shorter than input.
    # Let's cap summary max_length to ~0.7 * input_len, but not below 32.
    if input_len <= 0:
        return max_summary_tokens, min_summary_tokens

    dynamic_max = max(32, int(input_len * 0.7))
    dynamic_max = min(dynamic_max, max_summary_tokens)

    dynamic_min = min(min_summary_tokens, max(dynamic_max // 4, 8))

    if dynamic_min >= dynamic_max:
        dynamic_min = max(8, dynamic_max // 2)

    return dynamic_max, dynamic_min


# ------------------------------------------------------------------
#                    MAIN ABSTRACTIVE API
# ------------------------------------------------------------------

def generate_abstractive_summary(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    max_summary_tokens: int = 256,
    min_summary_tokens: int = 64,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
) -> str:
    """
    Main API used in main.py and evaluation scripts.

    - Supports multiple models (BART, PEGASUS, FLAN-T5, etc.)
    - Handles long inputs by:
        1) chunking input text
        2) summarizing each chunk
        3) doing summary-of-summaries if needed

    Args:
        text: raw input text
        model_name: HF model name string
        max_summary_tokens: upper bound on generated tokens
        min_summary_tokens: lower bound on generated tokens (if feasible)
        num_beams: beam search width
        no_repeat_ngram_size: anti-repetition constraint
        length_penalty: generation length penalty

    Returns:
        final_summary: a single abstractive summary string
    """
    text = (text or "").strip()
    if not text:
        return ""

    summarizer = _get_summarizer(model_name=model_name)

    # 1) Chunk long text
    chunks = _chunk_text(text, model_name=model_name)
    partial_summaries: List[str] = []

    # Detect if model is T5/FLAN-T5 style
    is_t5_like = _is_t5_like(model_name)

    # 2) Summarize each chunk
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue

        effective_input = ch
        if is_t5_like:
            # T5 / FLAN-T5 typically use a 'summarize:' prefix
            effective_input = "summarize: " + ch

        # adapt max/min lengths based on this chunk
        dyn_max_len, dyn_min_len = _adaptive_lengths(
            summarizer,
            effective_input,
            max_summary_tokens=max_summary_tokens,
            min_summary_tokens=min_summary_tokens,
        )

        out = summarizer(
            effective_input,
            max_length=dyn_max_len,
            min_length=dyn_min_len,
            do_sample=False,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
        )
        summary_text = out[0]["summary_text"].strip()
        partial_summaries.append(summary_text)

    if not partial_summaries:
        return ""

    if len(partial_summaries) == 1:
        # Only one chunk => done
        return partial_summaries[0]

    # 3) Summary-of-summaries
    combined = " ".join(partial_summaries)

    effective_input = combined
    if is_t5_like:
        effective_input = "summarize: " + combined

    dyn_max_len, dyn_min_len = _adaptive_lengths(
        summarizer,
        effective_input,
        max_summary_tokens=max_summary_tokens,
        min_summary_tokens=min_summary_tokens,
    )

    final = summarizer(
        effective_input,
        max_length=dyn_max_len,
        min_length=dyn_min_len,
        do_sample=False,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        length_penalty=length_penalty,
    )
    final_summary = final[0]["summary_text"].strip()
    return final_summary
