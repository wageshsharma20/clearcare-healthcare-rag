"""
init_db.py
=============================================================================
Knowledge Base Initialization & Verification Script.

Run this script to:
1. Ingest all medical guidelines from data/knowledge_base/
2. Chunk them with section metadata
3. Compute 384-dimensional dense embeddings
4. Index them into ChromaDB on disk
5. Run a self-test query to verify retrieval accuracy
=============================================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path so imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR
from src.chunker import MedicalChunker
from src.vector_store import VectorStoreManager


def build_and_verify_database():
    print("=" * 70)
    print("🏥 ClearCare AI — Knowledge Base Ingestion & Vector Indexing")
    print("=" * 70)

    # 1. Initialize Chunker
    chunker = MedicalChunker(chunk_size=700, chunk_overlap=120)
    print(f"Reading clinical guidelines from: {KNOWLEDGE_BASE_DIR}")

    chunks = chunker.load_directory(KNOWLEDGE_BASE_DIR)
    print(f"✓ Successfully generated {len(chunks)} chunks across guidelines.")

    # 2. Initialize Vector Store & reset for fresh index
    vector_store = VectorStoreManager(persist_dir=str(CHROMA_PERSIST_DIR))
    vector_store.reset()

    print(f"Computing embeddings and indexing into ChromaDB...")
    indexed_count = vector_store.add_chunks(chunks)
    print(f"✓ Successfully indexed {indexed_count} chunks into ChromaDB at {CHROMA_PERSIST_DIR}")

    # 3. Print Database Stats
    stats = vector_store.get_stats()
    print("\n--- Knowledge Base Statistics ---")
    print(f"Total Chunks in Vector DB : {stats['total_chunks']}")
    print(f"Embedding Model           : {stats['embedding_model']}")
    print("Indexed Guideline Sources:")
    for src in stats["unique_sources"]:
        print(f"  • {src}")

    # 4. Run a verification query
    test_query = "What causes elevated ALT or liver enzyme inflammation?"
    print(f"\n--- Testing Semantic Retrieval ---")
    print(f"Query: \"{test_query}\"")
    results = vector_store.similarity_search(test_query, top_k=2)

    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}]")
        print(f"Source: {res['source']} | Section: {res['section']}")
        print(f"Similarity Score: {res['similarity'] * 100:.1f}% (Distance: {res['distance']})")
        print(f"Excerpt: {res['text'][:180]}...")

    print("\n" + "=" * 70)
    print("🎉 Knowledge base is fully initialized and verified!")
    print("=" * 70)


if __name__ == "__main__":
    build_and_verify_database()
