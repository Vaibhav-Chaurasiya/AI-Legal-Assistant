# ⚖️ AI Legal Assistant Chatbot (GenAI + RAG)

This is a Python-based Legal Assistant chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer legal questions from the **Indian Penal Code (IPC)**.

It parses a legal PDF, chunks the text, embeds it into vectors using Hugging Face's `sentence-transformers`, stores it in a **FAISS** index, and uses a Hugging Face **LLM** (`HuggingFaceH4/zephyr-7b-beta`) to answer user queries.

## 🚀 Features

- 📄 Parses IPC legal PDF using PyMuPDF
- ✂️ Chunks long legal text into manageable pieces
- 🧠 Embeds chunks using `all-MiniLM-L6-v2` from Hugging Face
- 🔍 Searches relevant sections using FAISS
- 💬 Generates answers using `HuggingFaceH4/zephyr-7b-beta`
- ✅ Runs on Hugging Face Inference API

## 🧰 Tech Stack

| Component         | Library/Tool                     |
|------------------|----------------------------------|
| PDF Parsing      | `PyMuPDF` (fitz)                 |
| Text Chunking    | Custom Python function           |
| Embedding        | `sentence-transformers`          |
| Vector Search    | `faiss-cpu`                      |
| LLM              | `HuggingFaceH4/zephyr-7b-beta`           |
| Backend API      | `FastAPI`                        |
| Frontend UI      | `React` (for chatbot interface)  |

## 📁 Folder Structure

```
AI Legal Assistant/
├── backend/
│   └── app.py                    # FastAPI backend logic
├── data/
│   └── IPC.pdf                   # Indian Penal Code PDF
├── embeddings.index              # FAISS vector index
├── texts.npy                     # Saved chunks
├── utils/
│   ├── pdf_reader.py             # Extracts text from PDF
│   ├── chunker.py                # Splits text into chunks
│   └── embedder.py               # Creates embeddings
├── frontend/                     # React app for chatbot UI
├── ingest.py                     # Run once to process PDF
├── query_bot.py                  # Optional terminal chatbot
├── requirements.txt
└── README.md
```

## ✅ How to Run (Dev Setup)

### 1. 📦 Install Python Requirements

```bash
pip install -r requirements.txt
```

### 2. 📄 Prepare IPC PDF

Place your IPC PDF in `data/IPC.pdf` (text-based, not scanned).

### 3. 🧠 Ingest PDF

```bash
python ingest.py
```

### 4. 🚀 Start Backend

```bash
python -m uvicorn backend.app:app --reload
```

### 5. 💬 Start Frontend

```bash
cd frontend
npm install
npm start
```

### 6. 🧪 Chat with the Bot

Visit [http://localhost:3000](http://localhost:3000) and start asking legal questions.

## 🔐 Hugging Face API Token

Set your Hugging Face API token in a `.env` file in your root directory:

```
API_TOKEN=your_hf_token_here
```

Get yours from: https://huggingface.co/settings/tokens

## 📦 requirements.txt

```txt
faiss-cpu
numpy
requests
sentence-transformers
pymupdf
fastapi
uvicorn
python-dotenv
```
(Frontend dependencies are in `frontend/package.json`)

## ✨ Credits

Built with ❤️ by Vaibhav  
LLM powered by Hugging Face `HuggingFaceH4/zephyr-7b-beta`

## 🔜 What's Next?

- 💬 Improve UI styling in React
- 🧠 Add section reference highlighting
- 💾 Store Q&A history to SQLite or PostgreSQL
- ☁️ Deploy fullstack app using Render + Vercel