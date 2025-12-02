"""
translator.py — FAST + FULL MULTILINGUAL SUPPORT
- MBART50 translation with batching
- GPU support
- No summarization or trimming
- Accurate sentence-level translation
"""

import re
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

# Load tokenizer + model
_TOKENIZER = MBart50Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
_MODEL = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# GPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL.to(device)

# FULL language mapping for your dropdown
LANG_CODE_MAP = {
    "en": "en_XX",
    "hi": "hi_IN",
    "es": "es_XX",
    "fr": "fr_XX",
    "de": "de_DE",
    "pt": "pt_XX",
    "bn": "bn_IN",
    "mr": "mr_IN",
    "ta": "ta_IN",
    "te": "te_IN",
}


# -------------------------
# Split into sentences
# -------------------------
def split_into_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# -------------------------
# Batch MBART Translation
# -------------------------
@torch.inference_mode()
def translate_batch(sentences, src_code, tgt_code):
    """
    Translates the list of sentences in a single MBART call (very fast).
    """
    _TOKENIZER.src_lang = src_code

    inputs = _TOKENIZER(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(device)

    outputs = _MODEL.generate(
        **inputs,
        forced_bos_token_id=_TOKENIZER.lang_code_to_id[tgt_code],
        num_beams=1,  # fastest
        early_stopping=True
    )

    return [
        _TOKENIZER.decode(t, skip_special_tokens=True)
        for t in outputs
    ]


# -------------------------
# Main translation function
# -------------------------
def rewrite_text(text, src_lang="en", tgt_lang="en", task="translate"):
    """
    Always performs PURE 1:1 translation.
    No summarization.
    No compression.
    """

    if src_lang == tgt_lang:
        return text.strip()

    src_code = LANG_CODE_MAP.get(src_lang, None)
    tgt_code = LANG_CODE_MAP.get(tgt_lang, None)

    # Safety: if invalid lang code
    if src_code is None or tgt_code is None:
        return text.strip()

    sentences = split_into_sentences(text)
    if not sentences:
        return text.strip()

    translated = translate_batch(sentences, src_code, tgt_code)

    return " ".join(translated)
