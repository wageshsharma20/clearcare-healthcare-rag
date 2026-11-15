# 🩺 ClearCare AI: Evidence-Based Patient Health Literacy & Lab Report Translator

> **A Healthcare Retrieval-Augmented Generation (RAG) System** that translates complex lab results and clinical discharge notes into plain, compassionate, 8th-grade level English—grounded strictly in peer-reviewed clinical practice guidelines.

---

## 📑 Table of Contents
1. [The Real-World Healthcare Problem](#-the-real-world-healthcare-problem)
2. [Why Pure LLMs Fail & Why RAG is Essential](#-why-pure-llms-fail--why-rag-is-essential)
3. [System Architecture](#-system-architecture)
4. [First-Principles Concept Guide](#-first-principles-concept-guide)
5. [Codebase Walkthrough (File by File)](#-codebase-walkthrough-file-by-file)
6. [Quickstart Guide](#-quickstart-guide)
7. [Evaluation & Viva Questions](#-evaluation--viva-questions)

---

## 🏥 The Real-World Healthcare Problem
- **Low Health Literacy:** Over 80 million adults struggle to understand diagnostic lab tests (CBC, CMP, Lipid panel) and hospital discharge instructions.
- **Patient Anxiety & Misinterpretation:** When patients see technical flags like *"AST 52 U/L"* or *"Mild Hypokalemia"*, they search the web and encounter alarming worst-case diagnoses.
- **Post-Discharge Adverse Events:** 20% of patients experience an adverse event within 30 days of hospital discharge, predominantly driven by medication confusion (e.g., stopping blood thinners after cardiac stents).

---

## 🚫 Why Pure LLMs Fail & Why RAG is Essential
| Feature | Generic LLM (Raw ChatGPT / Gemini) | ClearCare RAG System |
| :--- | :--- | :--- |
| **Hallucination Risk** | High (Can invent normal ranges or catastrophic diagnoses) | **Zero / Minimized** (Grounded strictly in verified guidelines) |
| **Source Attribution** | Cannot cite exact hospital protocols | **100% Traceable** (Numbered citations with cosine scores) |
| **Out-of-Domain Boundaries**| Speculates when uncertain | Explicitly declines to guess if guideline match is low |
| **Cost & Latency** | High recurring API costs | Embeddings run **100% locally on CPU ($0)**; free Groq LLM |

---

## 🏗️ System Architecture

```
[Patient Lab Report / Discharge Summary]
                  │
                  ▼
     [Targeted Keyword & Entity Scanner] (ALT, LDL, Creatinine, Stent...)
                  │
                  ▼
         [ChromaDB Vector Store] <── Ingested Guidelines (MedlinePlus, ACC/AHA, AACC)
     (Dense 384-dim Cosine Similarity Search)
                  │
                  ▼
    [Top-K Clinical Chunks Filtered by Threshold >= 0.35]
                  │
                  ▼
       [Augmented Prompt with Medical Guardrails]
                  │
                  ▼
          [Groq LLM (Llama 3.3 70B)]
                  │
        ┌─────────┴────────────────────────┐
        ▼                                  ▼
[Plain-English Translation]        [Transparency Inspector]
• Traffic Light (🟢/🟡/🔴)         • Raw chunk text
• Reassuring explanations          • Cosine similarity percentages
• "Questions for Your Doctor"      • Exact source citations
```

---

## 🧠 First-Principles Concept Guide

### 1. Document Chunking & Overlap
- **Chunking (700 characters):** Dividing long medical texts into discrete conceptual blocks (e.g., one block specifically detailing *Serum Creatinine* and *eGFR*).
- **Chunk Overlap (120 characters):** Sliding window overlap ensures that qualification clauses (e.g., *"However, in pregnant patients, this elevation is physiological"*) are never severed at a boundary.

### 2. Dense Vector Embeddings
- Standard keyword search (TF-IDF, BM25) fails when terms differ (*"high blood sugar"* vs. *"hyperglycemia"*).
- We use `all-MiniLM-L6-v2` to map text into a **384-dimensional geometric coordinate space**.
- Semantically similar concepts cluster together in this high-dimensional space regardless of vocabulary differences.

### 3. Cosine Similarity & Distance
- Given query vector $\vec{q}$ and chunk vector $\vec{c}$:
  $$\text{Cosine Similarity} = \frac{\vec{q} \cdot \vec{c}}{\|\vec{q}\| \|\vec{c}\|}$$
- ChromaDB calculates cosine distance ($d = 1 - \text{similarity}$). We convert this to an intuitive percentage match:
  $$\text{Similarity \%} = (1.0 - d) \times 100$$

### 4. Grounded Prompt Engineering & Medical Guardrails
- **Reading Level Constraint:** Forced to output at 6th-to-8th grade reading level.
- **Categorization:** Classifies tests into 🟢 Normal, 🟡 Borderline, 🔴 Needs Discussion.
- **Doctor Questions Checklist:** Bridges patient-doctor communication.
- **Safety Disclaimer:** Explicitly disclaims formal diagnostic substitution.

---

## 📂 Codebase Walkthrough (File by File)

```text
healthcare-rag/
├── data/
│   ├── knowledge_base/               # Authoritative clinical guidelines (Markdown)
│   │   ├── cbc_blood_count.md        # WBC, RBC, Hemoglobin, Platelets ranges & causes
│   │   ├── metabolic_panel_cmp.md    # Glucose, BUN, Creatinine, eGFR, ALT, AST, Bilirubin
│   │   ├── lipid_cholesterol_panel.md# LDL, HDL, Triglycerides, HbA1c, ASCVD risk
│   │   ├── thyroid_tsh_panel.md      # TSH, Free T4, Hypo/Hyperthyroidism
│   │   └── discharge_care_protocols.md# Post-op care, DAPT cardiac stent warnings, red flags
│   └── sample_reports/               # Pre-loaded realistic patient cases
│       ├── sample_annual_checkup_labs.txt
│       ├── sample_cardiac_discharge.txt
│       └── sample_kidney_diabetic_labs.txt
├── src/
│   ├── config.py                     # Central constants, chunk sizes, model names
│   ├── chunker.py                    # Markdown & PDF parser with heading-aware chunking
│   ├── vector_store.py               # ChromaDB client & SentenceTransformer embeddings
│   ├── retriever.py                  # Clinical keyword extraction, multi-query & score filter
│   ├── rag_chain.py                  # Medical system prompt, Groq API adapter & fallback
│   └── init_db.py                    # Database creation & self-test script
├── app.py                            # Interactive Streamlit Web Application
├── requirements.txt                  # Python dependencies
└── README.md                         # This comprehensive manual
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```bash
cd healthcare-rag
source .venv/bin/activate
```

### 2. Verify Database (Optional)
```bash
python src/init_db.py
```

### 3. Run the Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Enter Free Groq API Key
1. Go to [Groq Console](https://console.groq.com/keys) and create a free API key.
2. Paste it in the left sidebar of the application.
3. Select any of the sample patient cases and click **"✨ Translate Report"**!

---

## 🎯 Evaluation & Viva Questions

### Q1: Why did you choose ChromaDB over storing embeddings in a standard SQL database?
**Answer:** Standard SQL databases index scalar values (numbers, text) via B-trees, which are designed for exact lookups or 1D range queries. Searching for nearest neighbors in 384 dimensions requires specialized spatial indexing like **HNSW (Hierarchical Navigable Small World)** graphs, which ChromaDB implements natively to provide sub-millisecond approximate nearest neighbor (ANN) retrieval.

### Q2: How does your application prevent medical hallucinations?
**Answer:** 
1. **Low Temperature:** We set temperature to 0.2 to discourage creative extrapolation.
2. **Strict System Prompt:** The LLM is instructed to rely solely on the retrieved chunks and explicitly decline to answer if evidence is absent.
3. **Similarity Threshold Filtering:** Any chunk with cosine similarity below 0.35 is discarded before reaching the LLM prompt.
4. **Mandatory Citations:** Every claim must reference `[Source X]`.

### Q3: What happens if a patient uploads a lab test with 15 different markers?
**Answer:** In `retriever.py`, our `ClinicalRetriever` runs an entity scanner across known biomarkers (e.g., Creatinine, ALT, LDL). It executes both a holistic query and targeted sub-queries for each identified marker, deduplicating the results. This ensures that a single test does not monopolize the vector search results.
