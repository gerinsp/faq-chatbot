---
title: FAQ Chatbot  
emoji: 🤖  
colorFrom: blue  
colorTo: purple  
sdk: gradio  
sdk_version: 3.50.2  
app_file: app.py  
pinned: false  
license: apache-2.0  
---

# Chatbot FAQ dengan RAG

## 🚀 Fitur Utama
- Menggunakan **LangChain + Vector Database (FAISS)**
- Menjalankan **chatbot secara lokal dengan Streamlit**
- Evaluasi performa menggunakan **Confusion Matrix, Accuracy, Precision, Recall, F1-score**

---

## 🔧 Instalasi
```bash
git clone https://github.com/username/chatbot-faq.git
cd chatbot-faq
python -m venv env
source env/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

---

## 📌 Cara Menjalankan

1. **Persiapkan dataset** di `data/faq.csv`
2. **Jalankan chatbot dengan Streamlit**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Evaluasi Performa
Jalankan skrip evaluasi chatbot:
```bash
python src/evaluation.py
```

---

## 📂 Struktur Proyek
```plaintext
chatbot-faq/
│── data/                   # Dataset FAQ
│── models/                 # Model (jika diperlukan)
│── src/                    # Kode utama chatbot
│── app.py                  # UI Streamlit
│── requirements.txt        # Dependensi proyek
│── README.md               # Dokumentasi proyek
```

---

## 📌 Teknologi yang Digunakan
- **LangChain** → pipeline NLP.
- **FAISS** → vector database untuk pencarian cepat.
- **Streamlit** → antarmuka chatbot.
- **Scikit-Learn** → evaluasi chatbot.

---

## 🎯 Kesimpulan
Chatbot FAQ berbasis **RAG + LangChain** ini bisa digunakan untuk customer support dan hanya menjawab berdasarkan **dataset CSV** yang diberikan. Sistem ini cocok untuk otomatisasi layanan pelanggan yang efisien.
