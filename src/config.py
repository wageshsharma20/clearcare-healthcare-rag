"""
config.py
=============================================================================
Central Configuration Module for ClearCare RAG Application.

Concepts to understand:
1. Pathlib: Provides cross-platform, robust file path operations.
2. Embedding Model: 'all-MiniLM-L6-v2' maps sentences to a 384-dimensional dense
   vector space. It is extremely fast (~14ms per sentence on CPU) and requires
   no API key or internet once downloaded.
3. Chunking Parameters:
   - CHUNK_SIZE: Character count per text block. 500-800 characters is optimal
     for discrete medical concepts (e.g., an individual biomarker definition).
   - CHUNK_OVERLAP: Preserves sentence context at borders so facts aren't split.
=============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from a .env file if present
load_dotenv()

# Base project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Local Embedding Model Settings (Runs 100% free on Mac CPU)
# 'all-MiniLM-L6-v2' produces 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 700        # Characters per chunk
DEFAULT_CHUNK_OVERLAP = 120     # Overlap characters to prevent mid-thought clipping

# Vector Database Settings
COLLECTION_NAME = "clearcare_medical_guidelines"
DEFAULT_TOP_K = 4               # Number of relevant chunks retrieved per query
SIMILARITY_THRESHOLD = 0.35     # Minimum cosine similarity to accept a chunk (0.0 to 1.0)

# Groq LLM Settings (Free API)
# Model options: 'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'gemma2-9b-it'
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
