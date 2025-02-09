from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import re
from src.config import GOOGLE_API_KEY
from src.database import load_vector_database

vector_store = load_vector_database()
qa = RetrievalQA.from_chain_type(
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        api_key=GOOGLE_API_KEY
    ),
    retriever=vector_store.as_retriever()
)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def get_answer(question):
    query = preprocess_text(question)
    response = qa.run(query)
    return response
