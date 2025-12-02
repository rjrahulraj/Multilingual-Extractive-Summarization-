# from utils.pdf_utils import extract_text_from_pdf
# from utils.text_utils import clean_and_split_sentences
# from models.embedding import train_word2vec, get_embeddings
# from models.clustering import cluster_sentences
# from models.dl_scorer import build_scoring_model
# from models.dl_scorer_ga import run_ga

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import EarlyStopping

# from collections import Counter, OrderedDict
# import numpy as np


# def get_sentence_embeddings(sentences):
#     """
#     Try to use sentence-transformers if available (better embeddings).
#     Fallback to project's Word2Vec train + get_embeddings if not installed.
#     """
#     try:
#         from sentence_transformers import SentenceTransformer
#         sbert = SentenceTransformer('all-MiniLM-L6-v2')
#         embs = sbert.encode(sentences, show_progress_bar=False)
#         return np.array(embs)
#     except Exception:
#         # Fallback: use existing project Word2Vec functions
#         w2v_model = train_word2vec(sentences)
#         embs = get_embeddings(sentences, w2v_model)
#         return np.array(embs)


# def safe_train_test_split(X, y, test_size=0.2, random_state=42):
#     """
#     Safe wrapper that disables stratify if any class has <2 samples.
#     """
#     class_counts = Counter(y)
#     if any(c < 2 for c in class_counts.values()):
#         stratify_param = None
#     else:
#         stratify_param = y

#     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)


# def order_clusters_by_first_appearance(cluster_labels):
#     """
#     Return cluster ids ordered by the index of their first sentence
#     """
#     unique_clusters = sorted(set(cluster_labels), key=lambda cid: np.min(np.where(cluster_labels == cid)))
#     return unique_clusters


# def dedupe_keep_order(sentences):
#     """
#     Remove duplicates while preserving order.
#     """
#     return list(OrderedDict.fromkeys(sentences))


# def summarize_pdf(
#     pdf_path=None,
#     custom_text=None,
#     clustering_methods=['agglomerative', 'kmeans', 'spectral', 'dbscan', 'birch'],
#     cluster_ratio=0.5,
#     epochs=50,
#     plot=False,
#     ga_params=None,
#     prefer_sbert=True
# ):
#     # ------------------------------
#     # Load text
#     # ------------------------------
#     if custom_text:
#         text = custom_text
#     else:
#         text = extract_text_from_pdf(pdf_path)

#     sentences = clean_and_split_sentences(text)
#     filtered_sentences = [s.strip() for s in sentences if 5 < len(s.split()) < 40]

#     if not filtered_sentences:
#         return []  # nothing to summarize

#     # ------------------------------
#     # Clustering setup
#     # ------------------------------
#     n_sentences = len(filtered_sentences)
#     # choose number of clusters relative to sentences (min 1, max n_sentences)
#     n_clusters = max(1, min(n_sentences, int(max(1, n_sentences * cluster_ratio))))

#     # ------------------------------
#     # Tokenization (for DL scorer)
#     # ------------------------------
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(filtered_sentences)
#     sequences = tokenizer.texts_to_sequences(filtered_sentences)
#     padded_seqs = pad_sequences(sequences, padding="post")

#     # ------------------------------
#     # Embeddings (SBERT preferred, else Word2Vec)
#     # ------------------------------
#     embeddings = get_sentence_embeddings(filtered_sentences)

#     # ------------------------------
#     # Create labels for optional supervised scorer
#     # Current project used a heuristic (sentence length). Keep it but improve:
#     # label = 1 if sentence length > 10 words OR it's among top central sentences
#     # This is only to let your DL scorer learn something; model is optional.
#     # ------------------------------
#     lengths = np.array([len(s.split()) for s in filtered_sentences])
#     # centrality via embeddings (higher means more central)
#     try:
#         sim_matrix = cosine_similarity(embeddings)
#         centrality = sim_matrix.sum(axis=1)
#     except Exception:
#         # fallback if embeddings cause issues
#         centrality = lengths.astype(float)

#     # normalize centrality
#     centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min() + 1e-9)
#     # label heuristic: either long sentence or high centrality
#     labels = np.array(((lengths > 10) | (centrality > np.percentile(centrality, 60))).astype(int))

#     # ------------------------------
#     # Safe train-test split
#     # ------------------------------
#     # If dataset too small, skip training and use unsupervised scoring only
#     trainable = n_sentences >= 6  # arbitrary small threshold to allow training
#     if trainable:
#         try:
#             X_train, X_val, y_train, y_val = safe_train_test_split(padded_seqs, labels)
#         except Exception:
#             trainable = False

#     # ------------------------------
#     # Build/train scoring model (optional)
#     # ------------------------------
#     model = None
#     model_scores = np.zeros(n_sentences, dtype=float)

#     if trainable:
#         vocab_size = len(tokenizer.word_index) + 1
#         input_length = padded_seqs.shape[1]
#         if ga_params is None:
#             ga_params = {
#                 "lstm_units": 64,
#                 "dropout_rate": 0.5,
#                 "lr": 0.001,
#                 "optimizer_name": "adam"
#             }
#         model = build_scoring_model(
#             vocab_size=vocab_size,
#             input_length=input_length,
#             lstm_units=ga_params.get("lstm_units", 64),
#             dropout_rate=ga_params.get("dropout_rate", 0.5),
#             lr=ga_params.get("lr", 0.001),
#             optimizer_name=ga_params.get("optimizer_name", "adam")
#         )

#         early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
#         # reduced verbosity and epochs to keep interactive apps responsive
#         model.fit(
#             X_train, y_train,
#             validation_data=(X_val, y_val),
#             epochs=epochs,
#             batch_size=16,
#             callbacks=[early_stop],
#             verbose=0
#         )

#         # validation check
#         try:
#             y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
#             val_acc = accuracy_score(y_val, y_val_pred)
#             print(f"Scorer validation accuracy: {val_acc:.4f}")
#         except Exception:
#             pass

#         # get model scores for all sentences
#         try:
#             model_scores = model.predict(padded_seqs).flatten()
#             # normalize
#             model_scores = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min() + 1e-9)
#         except Exception:
#             model_scores = np.zeros(n_sentences, dtype=float)

#     # ------------------------------
#     # Unsupervised centrality score (always available)
#     # ------------------------------
#     try:
#         sim_matrix = cosine_similarity(embeddings)
#         centrality_score = sim_matrix.sum(axis=1)
#         centrality_score = (centrality_score - centrality_score.min()) / (centrality_score.max() - centrality_score.min() + 1e-9)
#     except Exception:
#         centrality_score = (lengths - lengths.min()) / (lengths.max() - lengths.min() + 1e-9)

#     # ------------------------------
#     # Final combined score
#     # If model exists, weight it; else rely on centrality
#     # ------------------------------
#     if trainable:
#         final_scores = 0.6 * model_scores + 0.4 * centrality_score
#     else:
#         final_scores = centrality_score

#     # ------------------------------
#     # Multi-clustering – choose best sentence(s) per cluster, preserve doc order
#     # ------------------------------
#     combined_summary_sentences = []

#     for method in clustering_methods:
#         try:
#             cluster_labels = np.array(cluster_sentences(embeddings, method, n_clusters))
#         except Exception:
#             # if a clustering method fails, skip it
#             continue

#         # handle possible -1 or noise labels; we keep them as-is
#         cluster_order = order_clusters_by_first_appearance(cluster_labels)

#         for cluster_id in cluster_order:
#             cluster_idxs = np.where(cluster_labels == cluster_id)[0]
#             if cluster_idxs.size == 0:
#                 continue

#             # choose top sentence(s) within cluster by final_scores
#             # choose top 1; you may change to top 2 if you want longer summary
#             best_local_idx = cluster_idxs[np.argmax(final_scores[cluster_idxs])]
#             combined_summary_sentences.append((best_local_idx, filtered_sentences[best_local_idx]))

#     # If nothing selected (edge cases), fall back to top central sentences
#     if not combined_summary_sentences:
#         topk = min(3, n_sentences)
#         top_idxs = np.argsort(-final_scores)[:topk]
#         combined_summary_sentences = [(i, filtered_sentences[i]) for i in top_idxs]

#     # ------------------------------
#     # Deduplicate while preserving original order
#     # combined_summary_sentences currently is list of (idx, sentence)
#     # sort by idx (document order), remove duplicates
#     # ------------------------------
#     combined_summary_sentences.sort(key=lambda x: x[0])
#     ordered_sentences = [s for _, s in combined_summary_sentences]
#     final_sentences = dedupe_keep_order(ordered_sentences)

#     # Optionally: join into a paragraph or return list (keep list for flexibility)
#     return final_sentences


# # ------------------------------
# # MAIN (test-only)
# # ------------------------------
# if __name__ == "__main__":
#     pdf_file = "data/GenSumm_A_Joint_Framework_for_Multi-Task_Tweet_Classification_and_Summarization_Using_Sentiment_Analysis_and_Generative_Modelling.pdf"

#     try:
#         # run GA to find hyperparams (if you want). wrap in try/except to avoid blocking.
#         try:
#             best_hyperparams = run_ga()
#         except Exception:
#             best_hyperparams = None

#         summary = summarize_pdf(
#             pdf_path=pdf_file,
#             plot=True,
#             ga_params=best_hyperparams
#         )

#         print("\n📝 SUMMARY:\n")
#         for i, s in enumerate(summary, 1):
#             print(f"{i}. {s}\n")

#     except Exception as e:
#         print(f"❌ Error during summarization: {e}")


################################################################
###### NEW CODE BELOW - EMPTY MAIN FILE FOR MINOR PROJECT ######
################################################################

"""
main.py
- Extractive summarization + multilingual rewriting + sentiment
- Q&A over the full document using FAISS RAG
"""

from typing import Dict, Any

# Summarizer modules
from gensumm.gensumm_extractive import (
    extract_text,
    detect_sections,
    detect_language,
    embed_sentences,
    split_sentences,
    extractive_summary,
)

# Translation + sentiment
from gensumm.translator import rewrite_text
from gensumm.sentiment_module import analyze_sentiment

# FAISS RAG module
from gensumm.qa_module import (
    build_context_embeddings,
    retrieve_relevant,
    answer_question_generative,
)


def summarize_document_main(
    file_path: str,
    target_lang: str = "en",
    prefer_sbert: bool = True,
    do_ocr: bool = False,
    compression_ratio: float = 0.20,  
    max_cap: int = 40,               
    min_sentences: int = 3,           
) -> Dict[str, Any]:
 
    full_text = extract_text(file_path, do_ocr=do_ocr)

    sections = detect_sections(file_path, full_text)

    out_sections = []
    all_sent_scores = {"neg": 0.0, "neu": 0.0, "pos": 0.0}
    count = 0
  
    for head, content in sections:
        sents = split_sentences(content)

        if not sents:
            continue

        n = len(sents)


        target = int(round(n * compression_ratio))
        max_sent = max(min_sentences, min(max_cap, target))
        max_sent = min(max_sent, n)  

        embs = embed_sentences(sents, prefer_sbert=prefer_sbert)
        extractive = extractive_summary(
            sents,
            embs,
            max_sent=max_sent,
        )
  
        try:
            src_lang = detect_language(extractive)
        except Exception:
            src_lang = "unknown"

  
        task = "summarize" if src_lang == target_lang else "translate"
        rewritten = rewrite_text(
            extractive,
            src_lang=src_lang,
            tgt_lang=target_lang,
            task=task,
        )
  
        sent = analyze_sentiment(extractive)
        if sent:
            for k in all_sent_scores:
                all_sent_scores[k] += sent.get(k, 0.0)
            count += 1

        out_sections.append(
            {
                "heading": head,
                "src_lang": src_lang,
                "extractive": extractive,
                "rewritten": rewritten,
                "sentiment": sent,
            }
        )
 
    overall_sentiment = {
        k: (all_sent_scores[k] / count if count else None) for k in all_sent_scores
    }

    md = "\n".join(
        f"### {s['heading']}\n{s['rewritten']}\n" for s in out_sections
    ).strip()

    return {
        "summary": md,
        "sections": out_sections,
        "overall_sentiment": overall_sentiment,
    }

  

def answer_question_main(
    file_path: str,
    question: str,
    do_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Simple RAG-style QA over the full document.
    """

    if not question.strip():
        return {"answer": "Please enter a valid question."}

    full_text = extract_text(file_path, do_ocr=do_ocr)

    sentences = split_sentences(full_text)

    if not sentences:
        return {
            "question": question,
            "answer": "No readable text found in the document.",
            "context": "",
        }

    embeddings = build_context_embeddings(sentences)

    context = retrieve_relevant(sentences, embeddings, question, top_k=3)

    answer = answer_question_generative(question, context)

    return {
        "question": question,
        "context": context,
        "answer": answer,
    }

