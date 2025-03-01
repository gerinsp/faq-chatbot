import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from src.config import FAQ_CSV_PATH, HUGGINGFACEHUB_API_TOKEN
from src.rag_pipeline import get_answer

model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

def evaluate_faq_model():
    """Evaluasi model chatbot FAQ menggunakan cosine similarity."""
    df_test = pd.read_csv(FAQ_CSV_PATH, sep=None, engine="python")

    y_true = [ans.strip().lower() for ans in df_test["Answer"].tolist()]

    y_pred_gemini = [get_answer(q).strip().lower() for q in df_test["Question"].tolist()]

    y_pred_rag = [get_answer(q, use_rag=True)['result'].strip().lower() for q in df_test["Question"].tolist()]

    y_true_embed = model.embed_documents(y_true)
    y_pred_gemini_embed = model.embed_documents(y_pred_gemini)
    y_pred_rag_embed = model.embed_documents(y_pred_rag)

    similarity_gemini = np.array([
        cosine_similarity([true_emb], [pred_emb])[0, 0]
        for true_emb, pred_emb in zip(y_true_embed, y_pred_gemini_embed)
    ])

    similarity_rag = np.array([
        cosine_similarity([true_emb], [pred_emb])[0, 0]
        for true_emb, pred_emb in zip(y_true_embed, y_pred_rag_embed)
    ])

    threshold = 0.8
    y_eval_gemini = (similarity_gemini >= threshold).astype(int)
    y_eval_rag = (similarity_rag >= threshold).astype(int)

    metrics_gemini = {
        "Model": "Gemini",
        "Akurasi": accuracy_score(y_eval_gemini, np.ones_like(y_eval_gemini)),
        "Presisi": precision_score(y_eval_gemini, np.ones_like(y_eval_gemini), zero_division=1),
        "Recall": recall_score(y_eval_gemini, np.ones_like(y_eval_gemini), zero_division=1),
        "F1-score": f1_score(y_eval_gemini, np.ones_like(y_eval_gemini), zero_division=1),
    }

    metrics_rag = {
        "Model": "Gemini + RAG",
        "Akurasi": accuracy_score(y_eval_rag, np.ones_like(y_eval_rag)),
        "Presisi": precision_score(y_eval_rag, np.ones_like(y_eval_rag), zero_division=1),
        "Recall": recall_score(y_eval_rag, np.ones_like(y_eval_rag), zero_division=1),
        "F1-score": f1_score(y_eval_rag, np.ones_like(y_eval_rag), zero_division=1),
    }

    return pd.DataFrame([metrics_gemini, metrics_rag]), similarity_gemini, similarity_rag