# RAG-powered-Document-Q-A-system
A agent which uses RAG to get the precise answer you need to know!! No more hallucinations. 

Link to my notion - https://www.notion.so/RAG-powered-Document-Q-A-system-32a6f987f28580119fadeb162d52bbd3?source=copy_link

 HEAD
# 🧠 DocMind — RAG-Powered Document Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you upload documents and ask questions using natural language. Powered by GPT-4o-mini, ChromaDB, and text-embedding-3-small.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│  Upload Docs │ Chat Interface │ Source Citations │ Stats │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (REST)
┌──────────────────────▼──────────────────────────────────┐
│               FastAPI Backend (Port 8000)                │
│  /upload  │  /query  │  /query/stream  │  /documents     │
└──────┬────────────────────────┬────────────────────┬────┘
       │                        │                    │
┌──────▼──────┐    ┌────────────▼──────┐    ┌───────▼──────┐
│  Document   │    │    RAG Chain      │    │  ChromaDB    │
│  Processor  │    │  LangChain +      │    │  (Vector DB) │
│  PyPDF,     │    │  GPT-4o-mini      │    │  Persisted   │
│  Docx2txt   │    │  Streaming        │    │  Locally     │
└──────┬──────┘    └────────────▲──────┘    └──────▲───────┘
       │                        │                  │
       │           ┌────────────┴──────┐           │
       └──────────►│  Embeddings       ├───────────┘
                   │  text-embed-3-sm  │
                   └───────────────────┘
```

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Document Types** | PDF, TXT, DOCX |
| 🔍 **Retrieval** | Semantic similarity search via ChromaDB |
| 🤖 **LLM** | GPT-4o-mini (fast, cost-efficient) |
| 🧩 **Embeddings** | text-embedding-3-small (OpenAI) |
| 📡 **Streaming** | SSE token-by-token streaming endpoint |
| 🎯 **Scoped Q&A** | Query all docs or a single document |
| 🗑️ **Doc Management** | Upload, list, delete documents |
| 📎 **Citations** | Sources with page numbers & relevance scores |
| 🐳 **Docker** | Full stack containerized with Docker Compose |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Option 1: Local (without Docker)

**1. Clone & Setup**
```bash
git clone <repo>
cd rag-doc-qa
```

**2. Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start FastAPI
uvicorn main:app --reload --port 8000
```

**3. Frontend** (new terminal)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Access: `http://localhost:8501`

---

### Option 2: Docker Compose

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env

docker-compose up --build
```

- Frontend: `http://localhost:8501`
- Backend API docs: `http://localhost:8000/docs`

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + model info |
| `POST` | `/documents/upload` | Upload & index a document |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document |
| `POST` | `/query` | Ask a question (sync) |
| `POST` | `/query/stream` | Ask a question (streaming SSE) |
| `GET` | `/stats` | Vector store statistics |

### Example: Upload a document
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf"
```

### Example: Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "k": 5}'
```

---

## ⚙️ Configuration

Edit `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_RETRIEVED_DOCS` | `5` | Default k for retrieval |

---

## 🗂️ Project Structure

```
rag-doc-qa/
├── backend/
│   ├── main.py              # FastAPI app & routes
│   ├── config.py            # Settings (pydantic-settings)
│   ├── document_processor.py # Load & chunk documents
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── rag_chain.py         # LangChain RAG pipeline
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 How It Works

1. **Upload** — Document is loaded (PyPDF/Docx2txt), split into overlapping chunks, embedded with `text-embedding-3-small`, and stored in ChromaDB.
2. **Query** — User question is embedded, top-k similar chunks are retrieved via cosine similarity.
3. **Generate** — Retrieved context + question → GPT-4o-mini → answer with source citations.
=======
# RAG-powered-Document-Q-A-system
A agent which uses RAG to get the precise answer you need to know!! No more hallucinations.
