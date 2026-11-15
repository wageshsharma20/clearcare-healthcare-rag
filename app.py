"""
app.py
=============================================================================
ClearCare AI — Streamlit Interactive Web Application.

A patient-centered, evidence-grounded clinical decision support and health
literacy translation platform powered by RAG (ChromaDB + Groq LLM).
=============================================================================
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="ClearCare AI — Clinical Lab & Discharge Translator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    KNOWLEDGE_BASE_DIR,
    SAMPLE_REPORTS_DIR,
    CHROMA_PERSIST_DIR,
    DEFAULT_GROQ_MODEL,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    GROQ_API_KEY
)
from src.chunker import MedicalChunker
from src.vector_store import VectorStoreManager
from src.retriever import ClinicalRetriever
from src.rag_chain import ClearCareRAGChain


# =============================================================================
# Cached System Initialization
# =============================================================================
@st.cache_resource(show_spinner="Initializing Medical Vector Database & Embeddings...")
def get_rag_components():
    """
    Initializes and caches the VectorStore, Retriever, and RAG Chain.
    Streamlit caches this so the 384-dim embedding model is only loaded once.
    """
    vector_store = VectorStoreManager(persist_dir=str(CHROMA_PERSIST_DIR))
    
    # If the collection is empty, auto-ingest default guidelines
    if vector_store.collection.count() == 0:
        chunker = MedicalChunker()
        chunks = chunker.load_directory(KNOWLEDGE_BASE_DIR)
        vector_store.add_chunks(chunks)

    retriever = ClinicalRetriever(vector_store=vector_store)
    rag_chain = ClearCareRAGChain(retriever=retriever)
    return vector_store, retriever, rag_chain


vector_store, retriever, rag_chain = get_rag_components()


# =============================================================================
# Helper Functions
# =============================================================================
def load_sample_report(filename: str) -> str:
    """Reads a sample report text from data/sample_reports/."""
    file_path = SAMPLE_REPORTS_DIR / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Initialize session state for chat and inspection
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "last_raw_prompt" not in st.session_state:
    st.session_state.last_raw_prompt = ""
if "current_report_text" not in st.session_state:
    st.session_state.current_report_text = ""


# =============================================================================
# Sidebar Configuration & Controls
# =============================================================================
with st.sidebar:
    st.title("⚙️ RAG System Controls")

    # 1. Groq API Key Input
    st.subheader("1. LLM Authentication")
    user_api_key = st.text_input(
        "Groq API Key (Free)",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free instant key from console.groq.com/keys"
    )

    if user_api_key:
        rag_chain.set_api_key(user_api_key)
        st.success("✓ Groq API Key Connected", icon="🟢")
    else:
        st.warning("Running in Offline Preview mode. Enter free Groq key for live generation.", icon="⚠️")
        st.caption("[Get a free Groq API key here](https://console.groq.com/keys)")

    model_option = st.selectbox(
        "Groq Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0
    )
    rag_chain.model_name = model_option

    st.divider()

    # 2. Sample Report Selector
    st.subheader("2. Sample Patient Data")
    sample_choice = st.selectbox(
        "Quick-Load Sample Patient Case:",
        options=[
            "-- Select a Sample Case --",
            "Routine Annual Checkup (Elevated LDL & ALT)",
            "Post-PCI Cardiac Stent Discharge",
            "Kidney & Diabetic Monitoring (Elevated Creatinine)"
        ]
    )

    if st.button("📥 Load Selected Sample Report", use_container_width=True):
        if "Annual Checkup" in sample_choice:
            st.session_state.current_report_text = load_sample_report("sample_annual_checkup_labs.txt")
        elif "Cardiac" in sample_choice:
            st.session_state.current_report_text = load_sample_report("sample_cardiac_discharge.txt")
        elif "Kidney" in sample_choice:
            st.session_state.current_report_text = load_sample_report("sample_kidney_diabetic_labs.txt")
        st.rerun()

    st.divider()

    # 3. Retrieval Hyperparameter Tuning
    st.subheader("3. Retrieval Parameters")
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=2,
        max_value=8,
        value=DEFAULT_TOP_K,
        help="Number of most similar clinical excerpts retrieved from ChromaDB."
    )
    sim_threshold = st.slider(
        "Cosine Similarity Threshold",
        min_value=0.20,
        max_value=0.60,
        value=SIMILARITY_THRESHOLD,
        step=0.05,
        help="Minimum similarity required to pass a chunk to the LLM."
    )

    st.divider()

    # 4. Knowledge Base Summary
    st.subheader("4. Knowledge Base Status")
    db_stats = vector_store.get_stats()
    st.metric("Total Indexed Chunks", db_stats["total_chunks"])
    with st.expander("Active Clinical Guidelines"):
        for src in db_stats["unique_sources"]:
            st.markdown(f"- 📄 `{src}`")


# =============================================================================
# Main Application Header
# =============================================================================
st.markdown("""
# 🩺 ClearCare AI
### *Evidence-Based Patient Health Literacy & Clinical Document Translator*
""")

# Medical Disclaimer Banner
st.info(
    "⚠️ **Educational Clinical Decision Support:** ClearCare AI translates medical documentation into plain language "
    "using verified clinical guidelines (MedlinePlus, ACC/AHA, AACC). It does not provide formal medical diagnoses or modify prescriptions. "
    "Always consult your licensed healthcare provider.",
    icon="ℹ️"
)

# Tabs Navigation
tab_translate, tab_chat, tab_inspector, tab_concepts = st.tabs([
    "📄 Report Translator",
    "💬 Ask ClearCare (Q&A)",
    "🔍 RAG Engine Inspector",
    "📚 Concept Cheat Sheet"
])


# =============================================================================
# TAB 1: Report Translator
# =============================================================================
with tab_translate:
    st.markdown("#### Paste Patient Lab Report, Blood Test, or Hospital Discharge Summary")
    
    report_input = st.text_area(
        label="Medical Document Text",
        value=st.session_state.current_report_text,
        height=220,
        placeholder="Paste lab results (e.g. Glucose, Creatinine, Lipid Panel, Blood Count) or hospital discharge notes..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        translate_clicked = st.button("✨ Translate Report", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear Input"):
            st.session_state.current_report_text = ""
            st.rerun()

    if translate_clicked and report_input.strip():
        with st.spinner("Analyzing biomarkers and cross-referencing clinical guidelines..."):
            result = rag_chain.translate_report(
                report_text=report_input,
                top_k=top_k,
                similarity_threshold=sim_threshold
            )
            st.session_state.last_retrieved_chunks = result["retrieved_chunks"]
            st.session_state.last_raw_prompt = result["raw_prompt"]

            st.markdown("---")
            st.markdown(result["response"])

            # Citation badges drawer
            if result["retrieved_chunks"]:
                with st.expander(f"📚 Verified Evidence Sources Used ({len(result['retrieved_chunks'])} guidelines matched)"):
                    for i, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(
                            f"**[Source {i}] {chunk['source']}** — *{chunk['section']}* "
                            f"(Match: **{int(chunk['similarity']*100)}%**)"
                        )
                        st.caption(chunk["text"])
                        st.divider()


# =============================================================================
# TAB 2: Interactive Q&A (Ask ClearCare)
# =============================================================================
with tab_chat:
    st.markdown("#### Have questions about your results or medications?")
    st.caption("Ask specific questions like: *'What does elevated ALT mean?'*, *'Can I stop taking Plavix?'*, or *'What is a normal eGFR?'*")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type your medical question here..."):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching clinical guidelines..."):
                result = rag_chain.answer_question(
                    question=prompt,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k,
                    similarity_threshold=sim_threshold
                )
                response_text = result["response"]
                st.markdown(response_text)
                
                st.session_state.last_retrieved_chunks = result["retrieved_chunks"]
                st.session_state.last_raw_prompt = result["raw_prompt"]
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})


# =============================================================================
# TAB 3: RAG Engine Inspector (Transparency & Evaluation Drawer)
# =============================================================================
with tab_inspector:
    st.markdown("### 🔍 RAG Engine Transparency Inspector")
    st.markdown(
        "This drawer shows the inner workings of the RAG pipeline for the most recent query, "
        "including exact vector similarity scores and the augmented prompt."
    )

    if not st.session_state.last_retrieved_chunks:
        st.info("Run a report translation or ask a question in Tab 1 or Tab 2 to inspect retrieved vector chunks.")
    else:
        st.markdown(f"#### 1. Retrieved Vector Chunks ({len(st.session_state.last_retrieved_chunks)} items)")
        
        for idx, chunk in enumerate(st.session_state.last_retrieved_chunks, 1):
            col_meta, col_bar = st.columns([2, 3])
            with col_meta:
                st.markdown(f"**Chunk #{idx}: `{chunk['id']}`**")
                st.markdown(f"📄 Source: **{chunk['source']}**")
                st.markdown(f"📌 Section: *{chunk['section']}*")
            with col_bar:
                st.metric("Cosine Similarity Score", f"{chunk['similarity'] * 100:.1f}%")
                st.progress(float(chunk["similarity"]))

            with st.expander("View Full Chunk Text"):
                st.code(chunk["text"], language="markdown")
            st.divider()

        st.markdown("#### 2. Raw Augmented Prompt Sent to LLM")
        with st.expander("Inspect Full Augmented Prompt"):
            st.code(st.session_state.last_raw_prompt, language="text")


# =============================================================================
# TAB 4: Concept Cheat Sheet (For Viva / Evaluation)
# =============================================================================
with tab_concepts:
    st.markdown("### 🎓 Core RAG Concepts Explained (Viva & Interview Guide)")
    
    with st.expander("1. What is RAG and why is it essential in Healthcare?", expanded=True):
        st.markdown("""
        - **Standard LLM Problem:** LLMs suffer from hallucinations, lack up-to-date knowledge, and cannot cite specific clinical hospital protocols. An invented drug dosage can harm a patient.
        - **RAG Architecture:** RAG separates **knowledge storage** from **language generation**. The vector database stores verified clinical guidelines; the LLM merely reads the retrieved facts and articulates them in plain English.
        """)

    with st.expander("2. How does Document Chunking & Overlap work?"):
        st.markdown("""
        - **Chunking:** Large medical PDFs (100+ pages) are split into 500-800 character blocks so each block represents a focused medical concept (e.g., Creatinine range).
        - **Chunk Overlap (120 chars):** Ensures that sentences at the boundary are not cut in half, preserving critical qualifications (e.g., *"However, in pregnant patients..."*).
        """)

    with st.expander("3. What are Dense Vector Embeddings?"):
        st.markdown("""
        - We use `all-MiniLM-L6-v2`, a 6-layer Transformer that maps text into a **384-dimensional vector space**.
        - Semantic proximity: The vector for *"elevated glucose"* is mathematically close to *"hyperglycemia"*, allowing semantic search that keyword matching would miss.
        """)

    with st.expander("4. How does Cosine Similarity work?"):
        st.markdown("""
        - Given query vector $\\vec{q}$ and chunk vector $\\vec{c}$, cosine similarity measures the angle $\\theta$ between them:
          $$\\text{Cosine Similarity} = \\frac{\\vec{q} \\cdot \\vec{c}}{\\|\\vec{q}\\| \\|\\vec{c}\\|}$$
        - Score ranges from 0.0 (unrelated) to 1.0 (identical semantic meaning).
        """)

    with st.expander("5. What are Medical Safety Guardrails?"):
        st.markdown("""
        - **Threshold Filtering:** Discards chunks below 0.35 similarity to prevent hallucinating on bad context.
        - **Strict System Persona:** Banned from making absolute diagnoses; forced to suggest doctor follow-up.
        - **Traceable Attribution:** Requires `[Source X]` citations for every medical claim.
        """)
