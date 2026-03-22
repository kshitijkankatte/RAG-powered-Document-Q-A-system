import streamlit as st
import requests
import json
import time
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(120,80,255,0.15) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(0,200,200,0.10) 0%, transparent 60%);
    animation: drift 8s ease-in-out infinite alternate;
}
@keyframes drift { from { transform: translate(0,0); } to { transform: translate(30px,20px); } }
.main-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #fff;
    margin: 0;
    position: relative;
    z-index: 1;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    margin: 0.5rem 0 0;
    position: relative;
    z-index: 1;
}

/* Cards */
.stat-card {
    background: #1e1e2e;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.75rem;
}
.stat-card .number { font-size: 2rem; font-weight: 600; color: #a78bfa; }
.stat-card .label  { font-size: 0.78rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; }

.doc-card {
    background: #1a1a2e;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.doc-icon { font-size: 1.4rem; }
.doc-name { font-size: 0.88rem; font-weight: 500; color: #e2e8f0; }
.doc-meta { font-size: 0.73rem; color: rgba(255,255,255,0.35); }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.25rem;
    margin: 1rem 0 0.25rem auto;
    max-width: 80%;
    color: #fff;
    font-size: 0.95rem;
    line-height: 1.6;
}
.chat-assistant {
    background: #1e1e30;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.25rem;
    margin: 0.25rem auto 1rem 0;
    max-width: 85%;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Source badges */
.source-badge {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #a78bfa;
    margin: 0.2rem;
}

/* Upload area */
.stFileUploader { border-radius: 12px !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
}

/* Buttons */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(109,40,217,0.4) !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────

def get_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json() if r.ok else {"total_documents": 0, "total_chunks": 0}
    except:
        return {"total_documents": 0, "total_chunks": 0}


def get_documents():
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json() if r.ok else []
    except:
        return []


def check_backend():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.ok
    except:
        return False


def upload_file(uploaded_file) -> Optional[dict]:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        r = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=120)
        if r.ok:
            return r.json()
        else:
            st.error(f"Upload error: {r.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def query_rag(question: str, doc_id: Optional[str] = None, k: int = 5) -> Optional[dict]:
    try:
        payload = {"question": question, "doc_id": doc_id, "k": k}
        r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
        if r.ok:
            return r.json()
        else:
            st.error(f"Query error: {r.json().get('detail', 'Unknown')}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def delete_document(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
        return r.ok
    except:
        return False


# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None
if "selected_doc_name" not in st.session_state:
    st.session_state.selected_doc_name = "All Documents"


# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 DocMind</h1>
    <p>RAG-powered Document Intelligence · GPT-4o-mini · ChromaDB · text-embedding-3-small</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Backend status
    is_online = check_backend()
    status_color = "🟢" if is_online else "🔴"
    st.markdown(f"**Backend** {status_color} {'Connected' if is_online else 'Offline — start backend'}")
    st.markdown("---")

    # Stats
    stats = get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="number">{stats['total_documents']}</div>
            <div class="label">Docs</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="number">{stats['total_chunks']}</div>
            <div class="label">Chunks</div></div>""", unsafe_allow_html=True)

    st.markdown("### 📂 Upload Document")
    uploaded = st.file_uploader(
        "PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx", "doc"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⬆️ Index Document", use_container_width=True, type="primary"):
            with st.spinner("Processing & embedding..."):
                result = upload_file(uploaded)
            if result:
                st.success(result["message"])
                st.json({
                    "Chunks": result["total_chunks"],
                    "Characters": result["total_characters"],
                    "Avg Chunk Size": result["avg_chunk_size"],
                })
                st.rerun()

    st.markdown("### 📚 Indexed Documents")
    documents = get_documents()

    if not documents:
        st.caption("No documents indexed yet.")
    else:
        # All documents option
        all_selected = st.session_state.selected_doc_id is None
        if st.button(
            f"{'✅ ' if all_selected else ''}🌐 All Documents",
            use_container_width=True,
            key="all_docs",
        ):
            st.session_state.selected_doc_id = None
            st.session_state.selected_doc_name = "All Documents"

        for doc in documents:
            is_selected = st.session_state.selected_doc_id == doc["doc_id"]
            ext = doc["filename"].rsplit(".", 1)[-1].upper() if "." in doc["filename"] else "FILE"
            icon = {"PDF": "📕", "TXT": "📄", "DOCX": "📝", "DOC": "📝"}.get(ext, "📎")

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    f"{'✅ ' if is_selected else ''}{icon} {doc['filename'][:22]}{'…' if len(doc['filename'])>22 else ''}",
                    key=f"doc_{doc['doc_id']}",
                    use_container_width=True,
                ):
                    st.session_state.selected_doc_id = doc["doc_id"]
                    st.session_state.selected_doc_name = doc["filename"]
            with col_del:
                if st.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete document"):
                    if delete_document(doc["doc_id"]):
                        if st.session_state.selected_doc_id == doc["doc_id"]:
                            st.session_state.selected_doc_id = None
                            st.session_state.selected_doc_name = "All Documents"
                        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=5)

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Chat Interface ─────────────────────────────────────────────────────────────
scope_label = f"🔍 Scope: **{st.session_state.selected_doc_name}**"
st.markdown(scope_label)

# Display history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">💬 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} Source(s) Used", expanded=False):
                    for src in msg["sources"]:
                        page_info = f" · Page {src['page']}" if src.get("page") else ""
                        score_pct = f"{src['relevance_score']*100:.1f}%" if src.get("relevance_score") else ""
                        st.markdown(f"**{src['filename']}**{page_info} {f'· Relevance: {score_pct}' if score_pct else ''}")
                        st.caption(src.get("excerpt", ""))
                        st.divider()

# Input
st.markdown("<br>", unsafe_allow_html=True)
question = st.chat_input("Ask a question about your documents…")

if question:
    if not is_online:
        st.error("❌ Backend is offline. Please start the FastAPI server.")
    elif not documents and st.session_state.selected_doc_id is None:
        st.warning("⚠️ Please upload at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🔍 Searching & generating answer..."):
            result = query_rag(
                question=question,
                doc_id=st.session_state.selected_doc_id,
                k=top_k,
            )

        if result:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "retrieved_chunks": result.get("retrieved_chunks", 0),
            })

        st.rerun()
