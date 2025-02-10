import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from src.config import FAQ_CSV_PATH, VECTOR_DB_PATH, HUGGINGFACEHUB_API_TOKEN

def initialize_vector_database(csv_path=FAQ_CSV_PATH):
    """Buat database vektor dari CSV."""
    df = pd.read_csv(csv_path)
    texts = df["Question"].tolist()
    metadata = [{"answer": ans} for ans in df["Answer"].tolist()]

    embedding = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=HUGGINGFACEHUB_API_TOKEN,
    )
    vector_store = FAISS.from_texts(texts, embedding, metadatas=metadata)
    vector_store.save_local(VECTOR_DB_PATH)

def load_vector_database():
    """Memuat database FAISS."""
    embedding = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_key=HUGGINGFACEHUB_API_TOKEN,
    )
    return FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)