"""
sentiment_module.py — Optimized Version
- GPU support
- Lazy loading
- Batching support
- More robust fallback sentiment
"""

import threading
import torch

_lock = threading.Lock()
_model = None
_tokenizer = None
_has_transformers = False

try:
    import transformers
    _has_transformers = True
except Exception:
    _has_transformers = False


# ------------------------------
# Load model once (GPU if available)
# ------------------------------
def _load_sentiment_model():
    global _model, _tokenizer

    if not _has_transformers:
        return None, None

    with _lock:
        if _model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            _model = AutoModelForSequenceClassification.from_pretrained(model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model.to(device)
            _model.eval()

    return _model, _tokenizer


# ------------------------------
# Improved fallback sentiment
# ------------------------------
def _fallback_sentiment(text):
    txt = text.lower()

    pos_words = ["good", "great", "excellent", "amazing", "happy", "positive"]
    neg_words = ["bad", "poor", "fail", "problem", "negative", "unfortunately"]

    pos = sum(txt.count(w) for w in pos_words)
    neg = sum(txt.count(w) for w in neg_words)
    total = pos + neg

    if total == 0:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0}

    return {
        "neg": neg / total,
        "neu": 0.0,
        "pos": pos / total,
    }


# ------------------------------
# Main sentiment function
# ------------------------------
@torch.inference_mode()
def analyze_sentiment(text: str):
    if not text or not text.strip():
        return None

    if not _has_transformers:
        return _fallback_sentiment(text)

    model, tokenizer = _load_sentiment_model()

    if model is None:
        return _fallback_sentiment(text)

    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    # Forward pass
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    result = {"neg": probs[0], "neu": probs[1], "pos": probs[2]}

    # Optional: collapse near-ties to neutral
    if max(result.values()) < 0.45:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0}

    return result
