import streamlit as st
import pandas as pd
import os
from src.database import initialize_vector_database, load_vector_database
from src.rag_pipeline import get_answer
from src.config import FAQ_CSV_PATH, VECTOR_DB_PATH
from src.evaluation import evaluate_faq_model
import seaborn as sns
import matplotlib.pyplot as plt

st.title("💬 Chatbot FAQ dengan LangChain + Streamlit")

uploaded_file = st.file_uploader("📂 Upload file CSV berisi FAQ", type=["csv"])

if uploaded_file:
    with open(FAQ_CSV_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("🔄 Memproses database vektor...")
    initialize_vector_database(FAQ_CSV_PATH)
    st.success("✅ FAQ berhasil diproses! Chatbot siap digunakan.")

if os.path.exists(VECTOR_DB_PATH):
    st.write("✅ Database vektor ditemukan! Silakan mulai bertanya.")

    question = st.text_input("❓ Masukkan pertanyaan:")

    if question:
        answer = get_answer(question)
        st.write(f"💬 **Jawaban:** {answer['result']}")

st.title("📊 Evaluasi Chatbot FAQ")

if st.button("🔍 Evaluasi Model"):
    st.write("🔄 Evaluating...")

    # Panggil fungsi evaluasi
    report_df, similarity_gemini, similarity_rag = evaluate_faq_model()

    # Tampilkan classification report
    st.subheader("📌 Metrik Evaluasi")
    st.dataframe(report_df)

    # Visualisasi distribusi cosine similarity
    st.subheader("📊 Distribusi Cosine Similarity")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(similarity_gemini, bins=20, kde=True, color="blue", label="Gemini")
    sns.histplot(similarity_rag, bins=20, kde=True, color="green", label="Gemini + RAG")
    plt.axvline(0.8, color="red", linestyle="--", label="Threshold 0.8")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frekuensi")
    plt.title("Distribusi Cosine Similarity antara Ground Truth dan Prediksi")
    plt.legend()
    st.pyplot(fig)
