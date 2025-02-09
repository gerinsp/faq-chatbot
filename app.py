import streamlit as st
import pandas as pd
import os
from src.database import initialize_vector_database, load_vector_database
from src.rag_pipeline import get_answer
from src.config import FAQ_CSV_PATH, VECTOR_DB_PATH

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
        st.write(f"💬 **Jawaban:** {answer}")
