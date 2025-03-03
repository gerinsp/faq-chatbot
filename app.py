import streamlit as st
import os
from src.database import initialize_vector_database, load_vector_database
from src.rag_pipeline import get_answer
from src.config import FAQ_CSV_PATH, VECTOR_DB_PATH
from src.evaluation import evaluate_faq_model
import seaborn as sns
import matplotlib.pyplot as plt

st.title("💬 Chatbot FAQ with LangChain + Streamlit")

uploaded_file = st.file_uploader("📂 Upload a CSV file containing FAQs", type=["csv"])

if uploaded_file:
    with open(FAQ_CSV_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("🔄 Processing vector database...")
    initialize_vector_database(FAQ_CSV_PATH)
    st.success("✅ FAQ successfully processed! The chatbot is ready to use.")

if os.path.exists(VECTOR_DB_PATH):
    st.write("✅ Vector database found! You can start asking questions.")

    use_rag = st.toggle("🔄 Use RAG", value=True)

    question = st.text_input("❓ Enter your question:")

    if question:
        with st.spinner("🔄 Generating answer..."):
            answer = get_answer(question, use_rag)

        st.success("✅ Answer generated!")

        if isinstance(answer, dict):
            st.write(f"💬 **Answer:** {answer['result']}")
        else:
            st.write(f"💬 **Answer:** {answer}")

st.title("📊 FAQ Chatbot Evaluation")

if st.button("🔍 Evaluate Model"):
    st.write("🔄 Evaluating...")

    # Call the evaluation function
    report_df, similarity_gemini, similarity_rag = evaluate_faq_model()

    # Display classification report
    st.subheader("📌 Evaluation Metrics")
    st.dataframe(report_df)

    # Visualize cosine similarity distribution
    st.subheader("📊 Cosine Similarity Distribution")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(similarity_gemini, bins=20, kde=True, color="blue", label="Gemini")
    sns.histplot(similarity_rag, bins=20, kde=True, color="green", label="Gemini + RAG")
    plt.axvline(0.8, color="red", linestyle="--", label="Threshold 0.8")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Cosine Similarity Distribution between Ground Truth and Predictions")
    plt.legend()
    st.pyplot(fig)
