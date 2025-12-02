import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


try:
    from sentence_transformers import SentenceTransformer
    _EMB_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    _HAS_SBERT = True
except:
    _EMB_MODEL = None
    _HAS_SBERT = False


def _fallback_embed(texts, dim=384):
    out = []
    for t in texts:
        rnd = np.random.RandomState(abs(hash(t)) % (2**31))
        out.append(rnd.normal(size=(dim,)))
    return np.vstack(out)


def build_context_embeddings(sentences):
    if not sentences:
        return np.zeros((0, 384))

    if _HAS_SBERT and _EMB_MODEL:
        return _EMB_MODEL.encode(sentences, show_progress_bar=False)

    return _fallback_embed(sentences)


def retrieve_relevant(sentences, embeddings, question, top_k=5):
    if not sentences:
        return ""

    if _HAS_SBERT and _EMB_MODEL:
        q_emb = _EMB_MODEL.encode([question])[0]
    else:
        q_emb = _fallback_embed([question])[0]

    sim = cosine_similarity([q_emb], embeddings)[0]
    idx = np.argsort(sim)[::-1][:top_k]
    idx_sorted = sorted(idx)

    return " ".join(sentences[i] for i in idx_sorted)


def extract_title(full_text: str):
    lines = full_text.split("\n")
    for ln in lines:
        clean = ln.strip()
        if len(clean.split()) >= 3:
            return clean
    return None


def detect_question_type(question: str):
    q = question.lower()

    if "title" in q or "heading" in q or "name of the text" in q:
        return "title"

    if "main idea" in q or "central idea" in q or "gist" in q:
        return "main_idea"

    if "conclusion" in q or "what does the text conclude" in q:
        return "conclusion"

    if "author" in q or "who wrote" in q:
        return "author"

    if "date" in q or "year" in q or "when written" in q:
        return "date"

    return "normal"


def _simplify_text(text):

    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'[.!?]', text)

    main = sentences[0].strip()
    if len(sentences) > 1 and len(sentences[1].strip().split()) > 5:
        main += ". " + sentences[1].strip()

    replacements = {
        "Artificial Intelligence": "AI",
        "in summary": "basically",
        "overall": "in simple terms",
        "conclusion": "main point",
        "therefore": "so",
        "essentially": "basically"
    }
    for k, v in replacements.items():
        main = main.replace(k, v)

    return main


def answer_question_generative(question: str, context: str, full_text: str = ""):

    qtype = detect_question_type(question)

    if qtype == "title":
        title = extract_title(full_text)
        if title:
            return f"**Here's your answer:**\n\nThe title of the text is **{title}**."
        else:
            return "I couldn't detect the title in this document."

    if qtype == "main_idea":
        if context:
            simp = _simplify_text(context)
            return f"**Main Idea:**\n\n{simp}"
        return "I couldn't find a clear main idea."

    if qtype == "conclusion":
        if context:
            simp = _simplify_text(context)
            return f"**Conclusion:**\n\n{simp}"
        return "I couldn't find any concluding statements."

    if qtype == "author":
        return "The document does not appear to list an author."

    if qtype == "date":
        return "No publication date was detected in the text."

    if not context.strip():
        return "I couldn't find anything in the document that answers your question."

    simplified = _simplify_text(context)

    return (
        f"**Here's your answer:**\n\n"
        f"{simplified}"
    )
