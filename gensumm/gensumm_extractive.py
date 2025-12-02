import re
from typing import List, Tuple, Optional
import numpy as np
import os

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
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
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

def extract_text_pdf(path: str, do_ocr: bool = False) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for PDF parsing (pip install pdfplumber).")
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
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required for Word file parsing (pip install python-docx).")
    doc = Document(path)
    text = []
    for para in doc.paragraphs:
        if para.text.strip():
            text.append(para.text.strip())
    return "\n".join(text)

def extract_text_image(path: str) -> str:
    if not _HAS_OCR:
        raise RuntimeError("pytesseract + PIL are required for OCR on images.")
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
        raise ValueError(f"Unsupported file type: {ext}")

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
        if not s:
            continue
        words = s.split()
        if len(words) >= min_words:
            out.append(s)
    return out

def clean_heading(h: str) -> str:
    h = h.strip().replace(":", "")
    if len(h) < 2:
        return "General"
    return h

def detect_sections(file_path: Optional[str], full_text: str) -> List[Tuple[str, str]]:
    if pdfplumber is not None and file_path is not None and file_path.lower().endswith(".pdf"):
        try:
            lines = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    grouped = {}
                    for ch in page.chars:
                        top = round(ch.get("top", 0.0), 1)
                        grouped.setdefault(top, []).append(ch)
                    for top in sorted(grouped.keys()):
                        items = grouped[top]
                        txt = "".join(ch.get("text", "") for ch in items).strip()
                        if not txt:
                            continue
                        size = float(np.mean([float(ch.get("size", 0.0)) for ch in items]))
                        lines.append((txt, size))
            if lines:
                sizes = np.array([s for _, s in lines])
                threshold = float(np.median(sizes) + 0.5 * np.std(sizes))
                sections = []
                cur_head = "General"
                buf = []
                for txt, size in lines:
                    if size >= threshold and len(txt.split()) <= 12:
                        if buf:
                            sections.append((clean_heading(cur_head), "\n".join(buf).strip()))
                        cur_head = txt
                        buf = []
                    else:
                        buf.append(txt)
                if buf:
                    sections.append((clean_heading(cur_head), "\n".join(buf).strip()))
                if len(sections) >= 2:
                    return sections
        except Exception:
            pass
    parts = re.split(r"\n\s*\n+", full_text.strip())
    sections = []
    for p in parts:
        if len(p.strip()) < 50:
            continue
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        head = clean_heading(lines[0]) if lines else "General"
        body = "\n".join(lines[1:]) if len(lines) > 1 else p
        sections.append((head, body.strip()))
    return sections or [("General", full_text.strip())]

def embed_sentences(sentences: List[str], prefer_sbert: bool = True) -> np.ndarray:
    if prefer_sbert and _HAS_SBERT and SBERT_MODEL is not None:
        return SBERT_MODEL.encode(sentences, show_progress_bar=False)
    dim = 384
    out = []
    for s in sentences:
        rnd = np.random.RandomState(abs(hash(s)) % (2**31))
        out.append(rnd.normal(size=(dim,)))
    return np.vstack(out)

def extractive_summary(
    sentences: List[str],
    embeddings: np.ndarray,
    max_sent: int,
) -> str:
    n = len(sentences)
    if n == 0:
        return ""
    if n <= max_sent:
        return " ".join(sentences)
    k = min(max_sent, n)
    sim = cosine_similarity(embeddings)
    if _HAS_NX:
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                weight = 1.0 - float(sim[i, j])
                if weight <= 0:
                    weight = 0.000001
                G.add_edge(i, j, weight=weight)
        centrality = nx.harmonic_centrality(G, distance="weight")
        scores = np.array([centrality.get(i, 0.0) for i in range(n)])
    else:
        scores = sim.sum(axis=1)
    idxs = np.argsort(-scores)
    chosen = sorted(set(idxs[:k]))
    return " ".join([sentences[i] for i in chosen])

def summarize_document(
    file_path: str,
    prefer_sbert: bool = True,
    do_ocr: bool = False,
    compression_ratio: float = 0.20,
    max_cap: int = 40,
    min_sentences: int = 3,
) -> str:
    full_text = extract_text(file_path, do_ocr=do_ocr)
    sections = detect_sections(file_path, full_text)
    final_chunks = []
    for head, content in sections:
        sents = split_sentences(content, min_words=3)
        if not sents:
            continue
        n = len(sents)
        target = int(round(n * compression_ratio))
        max_sent = max(min_sentences, min(max_cap, target))
        max_sent = min(max_sent, n)
        embs = embed_sentences(sents, prefer_sbert=prefer_sbert)
        summ = extractive_summary(sents, embs, max_sent=max_sent)
        final_chunks.append(f"### {clean_heading(head)}\n{summ}\n")
    return "\n".join(final_chunks)
