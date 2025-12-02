import streamlit as st
import os

from main import (
    summarize_document_main,
    answer_question_main
)

st.set_page_config(page_title="GenSumm Multilingual Summarizer + Q&A", layout="wide")
st.title("📄 GenSumm — Multilingual Summarizer + Q&A")

uploaded = st.file_uploader(
    "Upload PDF, Word (.docx), or Image",
    type=["pdf", "docx", "png", "jpg", "jpeg"]
)

col1, col2, col3 = st.columns(3)
with col1:
    do_ocr = st.checkbox("Run OCR on pages/images", value=False)
with col2:
    prefer_sbert = st.checkbox("Prefer SBERT embeddings", value=True)
with col3:
    LANG_OPTIONS = {
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Portuguese": "pt",
        "Bengali": "bn",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
    }

    target_label = st.selectbox("Output language", options=list(LANG_OPTIONS.keys()))
    target_lang = LANG_OPTIONS[target_label]


if uploaded:
    if not os.path.exists("data"):
        os.makedirs("data")

    path = os.path.join("data", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())

    st.info(f"Uploaded file: {uploaded.name}")

    # ----- SUMMARY -----
    with st.spinner("Processing document..."):
        try:
            out = summarize_document_main(
                path,
                target_lang=target_lang,
                prefer_sbert=prefer_sbert,
                do_ocr=do_ocr
            )
        except Exception as e:
            st.error(f"Processing failed: {e}")
            raise

    st.success("Document processed successfully ✔")

    st.subheader("📝 Structured Summary")
    st.markdown(out["summary"], unsafe_allow_html=True)

    st.subheader("📊 Overall Sentiment")
    st.write(out["overall_sentiment"])

    st.subheader("📂 Per-section Details")
    for sec in out["sections"]:
        st.markdown(f"**{sec['heading']}** — Detected Language: `{sec['src_lang']}`")
        st.write("**Extractive Summary:**")
        st.write(sec["extractive"])
        st.write("**Translated:**")
        st.write(sec["rewritten"])
        st.write("**Sentiment:**")
        st.write(sec["sentiment"])
        st.write("---")

    # ----- Q&A -----
    st.header("🔍 Ask Questions About The Document (Q&A)")

    user_q = st.text_input("Enter your question about the document:")

    if st.button("Get Answer"):
        if not user_q.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("Searching document and generating answer..."):
                try:
                    qa = answer_question_main(path, user_q, do_ocr=do_ocr)
                except Exception as e:
                    st.error(f"Q&A failed: {e}")
                    raise

            st.subheader("🧠 Answer")
            st.write(qa["answer"])

            with st.expander("📌 Retrieved Context Used for Answering"):
                st.write(qa["context"])
