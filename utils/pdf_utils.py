# pdf_utils.py
import PyPDF2

def extract_text_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text += p.extract_text() or ""
    return text
