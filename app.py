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

# Custom Health-Tech Clinical Styling (Apple Health & Mayo Clinic Aesthetic)
st.markdown("""
<style>
    /* Global typography & clean clinical styling */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Clean tab bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 14px;
        color: #475569;
        border-radius: 8px 8px 0 0;
        padding: 0 16px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0284C7;
        border-bottom: 3px solid #0284C7;
        background: #F0F9FF;
    }
    
    /* Stage cards & clinical containers */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }
    .clinical-card:hover {
        border-color: #BAE6FD;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Stage badge */
    .stage-pill {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 9999px;
        background: #E0F2FE;
        color: #0369A1;
        border: 1px solid #BAE6FD;
        margin-bottom: 8px;
    }
    
    /* Metric pill */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        color: #334155;
    }
    
    /* Disclaimer banner */
    .disclaimer-banner {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 20px;
        color: #78350F;
        font-size: 13px;
        line-height: 1.5;
    }
    
    /* Traffic-light legend */
    .traffic-legend {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .traffic-pill-green {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .traffic-pill-yellow {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .traffic-pill-red {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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


# Initialize session state for chat, inspection, and results
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []
if "last_raw_prompt" not in st.session_state:
    st.session_state.last_raw_prompt = ""
if "current_report_text" not in st.session_state:
    st.session_state.current_report_text = ""
if "last_translation_output" not in st.session_state:
    st.session_state.last_translation_output = ""


# =============================================================================
# Sidebar Configuration & Controls
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System Controls")

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
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Chunks", db_stats["total_chunks"])
    with col_stat2:
        st.metric("Embed Dim", "384-d")
    
    with st.expander("Active Clinical Guidelines (5 Files)"):
        for src in db_stats["unique_sources"]:
            st.markdown(f"- 📄 `{src}`")


# =============================================================================
# Main Application Header
# =============================================================================
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown("""
    # 🩺 ClearCare AI
    ##### *Evidence-Grounded Patient Health Literacy & Clinical Document Translator*
    """)
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 15px;">
        <span class="metric-pill" style="color: #0284C7; border-color: #BAE6FD; background: #F0F9FF;">
            🟢 RAG Online
        </span>
    </div>
    """, unsafe_allow_html=True)

# Medical Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    <strong>ℹ️ Educational Clinical Decision Support:</strong> ClearCare AI translates complex clinical documentation into plain, empowering language using verified clinical guidelines (MedlinePlus, ACC/AHA, AACC). It does not provide formal medical diagnoses or alter prescriptions. Always consult your licensed healthcare provider.
</div>
""", unsafe_allow_html=True)

# Tabs Navigation (4 clean tabs — Concept Cheat Sheet removed, Pipeline Architecture added)
tab_translate, tab_chat, tab_inspector, tab_pipeline = st.tabs([
    "📄 Report Translator",
    "💬 Ask ClearCare (Q&A)",
    "🔍 RAG Engine Inspector",
    "🏗️ Project Pipeline & Architecture"
])


# =============================================================================
# TAB 1: Report Translator
# =============================================================================
with tab_translate:
    st.markdown("#### Paste Patient Lab Report, Blood Test, or Hospital Discharge Summary")
    
    # Quick Sample Pills for fast testing
    st.markdown("**Quick-Fill Sample Report:**")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("🧪 Annual Checkup (LDL & ALT)", use_container_width=True):
            st.session_state.current_report_text = load_sample_report("sample_annual_checkup_labs.txt")
            st.rerun()
    with col_s2:
        if st.button("❤️ Cardiac Stent Discharge", use_container_width=True):
            st.session_state.current_report_text = load_sample_report("sample_cardiac_discharge.txt")
            st.rerun()
    with col_s3:
        if st.button("🩸 Kidney & Diabetes Monitoring", use_container_width=True):
            st.session_state.current_report_text = load_sample_report("sample_kidney_diabetic_labs.txt")
            st.rerun()

    report_input = st.text_area(
        label="Medical Document Text",
        value=st.session_state.current_report_text,
        height=200,
        placeholder="Paste lab results (e.g. Glucose, Creatinine, Lipid Panel, Blood Count) or hospital discharge notes..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        translate_clicked = st.button("✨ Translate Report", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("Clear Input"):
            st.session_state.current_report_text = ""
            st.session_state.last_translation_output = ""
            st.rerun()

    if translate_clicked and report_input.strip():
        with st.spinner("Analyzing biomarkers, scanning clinical entities, and cross-referencing guidelines..."):
            result = rag_chain.translate_report(
                report_text=report_input,
                top_k=top_k,
                similarity_threshold=sim_threshold
            )
            st.session_state.last_retrieved_chunks = result["retrieved_chunks"]
            st.session_state.last_raw_prompt = result["raw_prompt"]
            st.session_state.last_translation_output = result["response"]

    # Render translation output if available
    if st.session_state.last_translation_output:
        st.markdown("---")
        
        # Action Bar with Copy Option
        col_out1, col_out2 = st.columns([4, 1])
        with col_out1:
            st.markdown("### 📋 Translated Patient Summary")
        with col_out2:
            st.caption("Hover over the code block below to copy full raw translation")

        st.markdown(st.session_state.last_translation_output)

        # Copyable raw text drawer
        with st.expander("📋 Copy Plain Text Translation (For Patient Handout)"):
            st.code(st.session_state.last_translation_output, language="markdown")

        # Citation badges drawer
        if st.session_state.last_retrieved_chunks:
            with st.expander(f"📚 Verified Evidence Sources Used ({len(st.session_state.last_retrieved_chunks)} guidelines matched)"):
                for i, chunk in enumerate(st.session_state.last_retrieved_chunks, 1):
                    match_pct = int(chunk['similarity'] * 100)
                    st.markdown(
                        f"**[Source {i}] {chunk['source']}** — *{chunk['section']}* "
                        f"(Match: **{match_pct}%**)"
                    )
                    st.caption(chunk["text"])
                    st.divider()


# =============================================================================
# TAB 2: Interactive Q&A (Ask ClearCare)
# =============================================================================
with tab_chat:
    st.markdown("#### Have questions about your results or medications?")
    st.caption("Ask specific clinical questions grounded strictly in our verified medical guidelines.")

    # Suggested Prompts Chips
    st.markdown("**Suggested Questions:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    quick_q = None
    with col_q1:
        if st.button("What is eGFR?", use_container_width=True):
            quick_q = "What does a normal eGFR level mean, and what causes it to decline?"
    with col_q2:
        if st.button("Why is ALT high?", use_container_width=True):
            quick_q = "What causes elevated ALT liver enzymes, and are there benign causes?"
    with col_q3:
        if st.button("Can I stop Plavix?", use_container_width=True):
            quick_q = "Can I stop taking Plavix or Aspirin after receiving a cardiac stent?"
    with col_q4:
        if st.button("LDL Cholesterol Goal", use_container_width=True):
            quick_q = "What are the recommended target LDL cholesterol levels according to ACC/AHA guidelines?"

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input (or triggered by quick prompt)
    prompt_to_send = quick_q or st.chat_input("Type your medical question here...")

    if prompt_to_send:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Generate RAG response
        with st.chat_message("assistant"):
            with st.spinner("Searching clinical guidelines..."):
                result = rag_chain.answer_question(
                    question=prompt_to_send,
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
        "This drawer reveals the inner mechanics of the RAG retrieval pipeline for the most recent query, "
        "including exact vector similarity scores, chunk metadata, and the raw prompt sent to the LLM."
    )

    if not st.session_state.last_retrieved_chunks:
        st.info("Run a report translation in Tab 1 or ask a question in Tab 2 to inspect retrieved vector chunks.")
    else:
        st.markdown(f"#### 1. Retrieved Vector Chunks ({len(st.session_state.last_retrieved_chunks)} items)")
        
        for idx, chunk in enumerate(st.session_state.last_retrieved_chunks, 1):
            col_meta, col_bar = st.columns([2, 3])
            with col_meta:
                st.markdown(f"**Chunk #{idx}: `{chunk['id']}`**")
                st.markdown(f"📄 Source: **{chunk['source']}**")
                st.markdown(f"📌 Section: *{chunk['section']}*")
            with col_bar:
                sim_score = float(chunk["similarity"])
                st.metric("Cosine Similarity Score", f"{sim_score * 100:.1f}%")
                st.progress(sim_score)

            with st.expander("View Full Chunk Text"):
                st.code(chunk["text"], language="markdown")
            st.divider()

        st.markdown("#### 2. Raw Augmented Prompt Sent to LLM")
        with st.expander("Inspect Full Augmented Prompt"):
            st.code(st.session_state.last_raw_prompt, language="text")


# =============================================================================
# TAB 4: Project Pipeline & Architecture (8-Stage Deep Dive)
# =============================================================================
with tab_pipeline:
    st.markdown("### 🏗️ Project Pipeline & Architecture")
    st.markdown(
        "ClearCare AI is engineered as an end-to-end, multi-stage clinical RAG system designed to "
        "eliminate hallucinations, overcome multi-analyte **Vector Blur**, and deliver evidence-grounded "
        "health literacy translations across **8 deterministic stages**."
    )

    # Visual Architecture Overview Flowchart
    st.markdown("#### 🔄 End-to-End System Architecture Overview")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph INGESTION ["Offline Ingestion Pipeline"]
            S1["1. Medical Ingestion<br/>(5 Clinical Guidelines)"] --> S2["2. Heading-Aware Chunking<br/>(700 chars + 120 overlap)"]
            S2 --> S3["3. Dense Embeddings<br/>(all-MiniLM-L6-v2, 384-d)"]
            S3 --> S4[("4. Vector Storage<br/>(ChromaDB + HNSW Index)")]
        end

        subgraph INFERENCE ["Online Clinical Inference Pipeline"]
            User["Patient Document / Query"] --> S5["5. Entity Scanning &<br/>Multi-Query Expansion"]
            S5 --> S4
            S4 --> S6["6. Cosine Thresholding<br/>(>= 0.35 & Source Tagging)"]
            S6 --> S7["7. Guardrailed Generation<br/>(Groq LPU + Llama 3.3 70B)"]
            S7 --> S8["8. Patient Output<br/>(Traffic Lights + Doctor Checklist)"]
        end

        style S1 fill:#E0F2FE,stroke:#0284C7,color:#0F172A
        style S2 fill:#E0F2FE,stroke:#0284C7,color:#0F172A
        style S3 fill:#E0F2FE,stroke:#0284C7,color:#0F172A
        style S4 fill:#DCFCE7,stroke:#16A34A,color:#0F172A
        style S5 fill:#FEF3C7,stroke:#D97706,color:#0F172A
        style S6 fill:#FEF3C7,stroke:#D97706,color:#0F172A
        style S7 fill:#F1F5F9,stroke:#475569,color:#0F172A
        style S8 fill:#DCFCE7,stroke:#16A34A,color:#0F172A
    ```
    """)

    st.markdown("---")
    st.markdown("#### 🔬 Detailed 8-Stage Pipeline Breakdown")

    # Interactive Stage Selector
    selected_stage = st.selectbox(
        "Jump to Stage Details:",
        [
            "All 8 Stages Overview",
            "Stage 1: Knowledge Ingestion (5 Guidelines)",
            "Stage 2: Heading-Aware Chunking (700 chars / 120 overlap)",
            "Stage 3: Local Dense Embeddings (all-MiniLM-L6-v2)",
            "Stage 4: Persistent Vector Storage (ChromaDB + HNSW)",
            "Stage 5: Clinical Entity Scanning & Multi-Query Expansion",
            "Stage 6: Similarity Thresholding & Context Formatting (>= 0.35)",
            "Stage 7: Guardrailed Generation (Groq LPU + Llama 3.3 70B)",
            "Stage 8: Patient-Centered Output (Traffic Lights & Checklist)"
        ]
    )

    # Stage 1 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 1: Knowledge Ingestion (5 Guidelines)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 01 &bull; INGESTION</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">📚 Knowledge Ingestion (5 Authoritative Guidelines)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Rather than relying on ungrounded pretraining memory, ClearCare indexes curated, peer-reviewed clinical guidelines from the 
                    <strong>American College of Cardiology (ACC)</strong>, <strong>American Heart Association (AHA)</strong>, 
                    <strong>American Association for Clinical Chemistry (AACC)</strong>, and <strong>NIH MedlinePlus</strong>.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">🧪 Comprehensive Metabolic Panel (CMP)</span>
                    <span class="metric-pill">🩸 Complete Blood Count (CBC)</span>
                    <span class="metric-pill">🥑 Lipid Cholesterol Panel</span>
                    <span class="metric-pill">🦋 Thyroid TSH Function</span>
                    <span class="metric-pill">❤️ Post-PCI Cardiac Stent Discharge</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Explore Ingestion Standards & Authority Sources"):
                st.markdown("""
                - **CMP Guidelines (`metabolic_panel_cmp.md`):** Reference intervals for Renal function (BUN, Creatinine, eGFR), Hepatic enzymes (ALT, AST, ALP, Bilirubin), and Electrolytes (Na+, K+, Ca2+).
                - **CBC Guidelines (`cbc_blood_count.md`):** White blood cells, differential, red blood cells, hemoglobin, hematocrit, and platelet counts.
                - **Lipid Guidelines (`lipid_cholesterol_panel.md`):** Total Cholesterol, LDL-C thresholds (<100 desirable, <70 high-risk), HDL-C, Triglycerides.
                - **Thyroid Guidelines (`thyroid_tsh_panel.md`):** TSH (0.4 - 4.0 mIU/L), Free T4, Free T3, differentiating subclinical vs overt thyroid dysfunction.
                - **Post-PCI Protocols (`discharge_care_protocols.md`):** Dual Antiplatelet Therapy (DAPT) compliance (Aspirin + Plavix), groin puncture site monitoring, and emergency red-flag symptoms.
                """)

    # Stage 2 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 2: Heading-Aware Chunking (700 chars / 120 overlap)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 02 &bull; PREPROCESSING</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">✂️ Heading-Aware Chunking (700 Chars + 120 Overlap)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Medical documents possess strict hierarchical context. A naive character splitter severs critical warnings from their biomarker definitions.
                    ClearCare uses a <strong>heading-aware sliding window chunker</strong> that prepends active section headers to every chunk.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Chunk Size: 700 chars</span>
                    <span class="metric-pill">Sliding Overlap: 120 chars</span>
                    <span class="metric-pill">Context Preservation: 100% Header Anchoring</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Why 700 Characters & 120 Overlap?"):
                st.markdown("""
                - **Why 700 chars?** Optimal granularity for discrete clinical tests (e.g., standard creatinine definition + reference range + benign causes). Prevents *context dilution* common in 2,000+ token chunking.
                - **Why 120 char overlap?** Prevents boundary clipping of critical contraindications (e.g., *"However, in pregnant patients..."* or *"Must not be discontinued without cardiologist approval"*).
                - **Heading Propagation:** Every chunk begins with `# [Document > Section > Sub-section]`, ensuring isolated chunks retain semantic context during dense retrieval.
                """)

    # Stage 3 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 3: Local Dense Embeddings (all-MiniLM-L6-v2)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 03 &bull; VECTOR EMBEDDINGS</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">🧠 Local Dense Embeddings (all-MiniLM-L6-v2, 384 Dimensions)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Converts text chunks into dense <strong>384-dimensional mathematical vectors</strong> where semantically synonymous medical concepts 
                    cluster together, overcoming the limitations of exact keyword matching.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Model: sentence-transformers/all-MiniLM-L6-v2</span>
                    <span class="metric-pill">Vector Space: 384 dimensions</span>
                    <span class="metric-pill">Latency: ~14ms on local CPU</span>
                    <span class="metric-pill">Privacy: 100% Offline (Zero PHI Egress)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Semantic Proximity in 384-D Vector Space"):
                st.markdown("""
                - **Synonym Resolution:** Vector representations map patient colloquialisms directly to medical guidelines:
                  - *"high blood sugar"* $\\approx$ *"elevated serum glucose"* $\\approx$ *"hyperglycemia"*
                  - *"kidney filtration"* $\\approx$ *"estimated glomerular filtration rate (eGFR)"*
                  - *"stent blood thinner"* $\\approx$ *"clopidogrel / Plavix DAPT compliance"*
                - **Local Privacy:** Runs 100% on the local CPU without transmitting Protected Health Information (PHI) to 3rd-party embedding APIs.
                """)

    # Stage 4 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 4: Persistent Vector Storage (ChromaDB + HNSW)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 04 &bull; VECTOR STORAGE</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">🗄️ Persistent Vector Storage (ChromaDB + HNSW Indexing)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Stores chunk embeddings in an on-disk ChromaDB collection indexed with <strong>Hierarchical Navigable Small World (HNSW)</strong> graphs, 
                    enabling sub-millisecond approximate nearest neighbor (ANN) retrieval at logarithmic complexity.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Engine: ChromaDB (Local SQLite/Parquet)</span>
                    <span class="metric-pill">Indexing: HNSW Graph (Cosine Distance)</span>
                    <span class="metric-pill">Query Complexity: O(log N)</span>
                    <span class="metric-pill">Metadata: Source, Section, Chunk ID</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("HNSW Graph Indexing Explained"):
                st.markdown("""
                - **Hierarchical Graph Navigation:** HNSW constructs a multi-layer graph where upper layers have long-range links for fast exploration and lower layers contain dense clusters for fine-grained nearest neighbor search.
                - **Persistence:** All embeddings and metadata are stored in `chroma_db/`, avoiding the cost of re-embedding on every application launch.
                - **Metadata Anchoring:** Every node stores rich metadata (`source`, `section`, `chunk_id`, `char_length`), allowing structured citation generation.
                """)

    # Stage 5 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 5: Clinical Entity Scanning & Multi-Query Expansion"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 05 &bull; CLINICAL RETRIEVAL</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">🔎 Clinical Entity Scanning & Multi-Query Expansion</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Overcomes the notorious <strong>"Vector Blur"</strong> problem. When a patient report contains 15+ different lab values, 
                    embedding the whole document creates a diluted average vector. ClearCare scans 30+ clinical biomarker entities, 
                    executes targeted sub-queries, and aggregates relevant candidate guidelines.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Entity Scanner: 30+ Biomarkers & Discharge Markers</span>
                    <span class="metric-pill">Strategy: Multi-Pass Sub-Queries</span>
                    <span class="metric-pill">Deduplication: Union of Candidates</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("The Multi-Analyte 'Vector Blur' Problem Solved"):
                st.markdown("""
                - **The Naive RAG Failure Mode:** In a report containing Glucose 112, ALT 52, LDL 142, and Platelets 210, a naive single vector embedding lands in the generic center of the vector space, retrieving vague overview chunks and missing specific liver enzyme or lipid guidelines.
                - **ClearCare's Solution:** 
                  1. Scans query for known clinical entities (Glucose, ALT, AST, LDL, Creatinine, eGFR, Stent, Plavix...).
                  2. If multiple entities exist, issues targeted sub-queries for each detected analyte.
                  3. Combines, deduplicates, and re-ranks candidate chunks by cosine similarity.
                """)

    # Stage 6 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 6: Similarity Thresholding & Context Formatting (>= 0.35)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 06 &bull; SAFETY FILTERING</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">🎯 Similarity Thresholding & Context Formatting (>= 0.35)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Acts as the primary mathematical defense against hallucinations. In medicine, returning an irrelevant chunk is worse 
                    than returning nothing. Chunks below <strong>0.35 cosine similarity</strong> are strictly discarded, and qualifying chunks 
                    are formatted with immutable <code>[Source X]</code> metadata anchors.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Cosine Threshold: >= 0.35 (Cosine Distance <= 0.65)</span>
                    <span class="metric-pill">Hallucination Defense: Zero Irrelevant Context</span>
                    <span class="metric-pill">Formatting: Standardized [Source X] Tagging</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Mathematical Thresholding & Prompt Construction"):
                st.markdown("""
                - **Cosine Similarity Equation:**
                  $$\\text{Cosine Similarity} = \\frac{\\mathbf{q} \\cdot \\mathbf{c}}{\\|\\mathbf{q}\\| \\|\\mathbf{c}\\|} = 1 - \\text{Cosine Distance}$$
                - **Hallucination Firewall:** If all retrieved chunks score below 0.35, ClearCare halts generation and explicitly notifies the patient that no verified clinical guideline matches the query, rather than synthesizing speculative text.
                - **Context Assembly:** Qualifying chunks are injected as:
                  ```text
                  [Source 1: metabolic_panel_cmp.md | Section: Kidney Function | Match: 84%]
                  Creatinine reference range: 0.6 - 1.3 mg/dL...
                  ```
                """)

    # Stage 7 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 7: Guardrailed Generation (Groq LPU + Llama 3.3 70B)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 07 &bull; LLM INFERENCE</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">⚡ Guardrailed Generation (Groq LPUs + Llama 3.3 70B at temp 0.2)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Powered by ultra-fast Groq LPUs running <strong>Llama 3.3 70B Versatile</strong> at a deterministic temperature of <strong>0.2</strong>. 
                    Enforces strict clinical guardrails: absolute diagnosis bans, mandatory benign causes, and mandatory citation attribution.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Model: Llama 3.3 70B Versatile</span>
                    <span class="metric-pill">Temperature: 0.2 (Deterministic & Grounded)</span>
                    <span class="metric-pill">Inference Speed: ~500 tokens/sec on Groq LPUs</span>
                    <span class="metric-pill">Guardrail: Zero Ungrounded Assertions</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Clinical Safety Rules Enforced in System Prompt"):
                st.markdown("""
                1. **No Absolute Diagnoses:** Prohibited from stating *"You have kidney disease"*; must state *"Your creatinine is slightly above reference ranges, which warrants follow-up discussion with your physician"*.
                2. **Mandatory Benign Causes:** Must explain benign explanations first (e.g. intense exercise, mild dehydration, dietary changes) before discussing clinical concerns to prevent health anxiety.
                3. **Strict Citation Attribution:** Every medical statement must be explicitly footnoted with its corresponding `[Source X]`.
                4. **Medication Safety Protection:** Strictly prohibited from suggesting medication discontinuation (e.g., dual antiplatelet therapy post-stent).
                """)

    # Stage 8 Card
    if selected_stage in ["All 8 Stages Overview", "Stage 8: Patient-Centered Output (Traffic Lights & Checklist)"]:
        with st.container():
            st.markdown("""
            <div class="clinical-card">
                <span class="stage-pill">STAGE 08 &bull; PATIENT EXPERIENCE</span>
                <h4 style="color: #0F172A; margin-top: 4px; margin-bottom: 8px;">🌟 Patient-Centered Output (Traffic-Light Badges & Doctor Checklist)</h4>
                <p style="color: #475569; font-size: 14px; line-height: 1.6;">
                    Translates opaque clinical jargon into reassuring, 8th-grade level English structured into executive summaries, 
                    intuitive traffic-light findings tables, and an empowering <strong>"Questions to Ask Your Doctor"</strong> checklist.
                </p>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
                    <span class="metric-pill">Readability: 8th-Grade Plain English</span>
                    <span class="metric-pill">Visuals: 🟢 Normal &bull; 🟡 Borderline &bull; 🔴 Attention</span>
                    <span class="metric-pill">Empowerment: Doctor Discussion Checklist</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Output Schema & Health Literacy Standards"):
                st.markdown("""
                - **Executive Summary:** A 2-to-3 sentence calming overview explaining what the report covers in plain words.
                - **Traffic-Light Findings Table:**
                  - 🟢 **Normal:** Result falls within standard reference intervals.
                  - 🟡 **Borderline:** Mild variance often related to hydration, diet, or timing.
                  - 🔴 **Needs Attention:** Out-of-range value requiring physician review.
                - **Plain-English Explanations:** What each test does, why it matters, and what benign factors can cause fluctuations.
                - **Doctor Discussion Checklist:** Concrete, bulleted questions for the patient to ask at their next clinical appointment.
                """)

    st.markdown("---")
    # Architecture Comparison Table
    st.markdown("#### 📊 Architecture Comparison: ClearCare RAG vs Alternatives")
    st.markdown("""
    | Architectural Dimension | Naive LLM (e.g. ChatGPT) | Fine-Tuned Medical LLM | ClearCare AI (Multi-Stage RAG) |
    | :--- | :--- | :--- | :--- |
    | **Hallucination Risk** | High (invents reference ranges) | Medium (memorization drift) | **Zero (strictly grounded in retrieved chunks)** |
    | **Auditability & Citations** | None (black-box generation) | None (weights cannot cite text) | **100% Traceable (`[Source X]` citations)** |
    | **Guideline Updating** | Requires full model retraining | Requires expensive fine-tuning | **Instant (drop new `.md` into knowledge base)** |
    | **Multi-Analyte Lab Reports** | Suffers from vector dilution | Confuses cross-analyte interactions | **Resolved via Entity Scanning & Multi-Query** |
    | **Data Privacy (PHI)** | Sends all data to public APIs | Sends all data to training clusters | **Local 384-d embeddings + private inference** |
    """)
