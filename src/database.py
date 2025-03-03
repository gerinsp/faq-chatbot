import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from src.config import FAQ_CSV_PATH, VECTOR_DB_PATH, HUGGINGFACEHUB_API_TOKEN


def initialize_vector_database(csv_path=FAQ_CSV_PATH):
    """Buat database vektor dari CSV dengan format yang benar."""

    df = pd.read_csv(csv_path, sep=None, engine='python')

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV harus memiliki kolom 'Question' dan 'Answer'.")

    texts = [f"Pertanyaan: {q}\nJawaban: {a}" for q, a in zip(df["question"], df["answer"])]

    embedding = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=HUGGINGFACEHUB_API_TOKEN,
    )

    vector_store = FAISS.from_texts(texts, embedding)
    vector_store.save_local(VECTOR_DB_PATH)

    print("✅ Database vektor berhasil dibuat dan disimpan!")


def load_vector_database():
    """Memuat database FAISS."""
    embedding = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=HUGGINGFACEHUB_API_TOKEN,
    )
    return FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)