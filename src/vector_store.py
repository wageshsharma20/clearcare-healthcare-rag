"""
vector_store.py
=============================================================================
Vector Database & Embedding Storage Engine using ChromaDB + SentenceTransformers.

Core Concepts to Master:
1. Dense Vector Embeddings:
   - Computers cannot directly understand that "myocardial infarction" and "heart attack"
     mean the exact same thing.
   - The SentenceTransformer model maps words into a 384-dimensional continuous
     geometric space where distance represents conceptual similarity.

2. Cosine Distance vs Cosine Similarity:
   - ChromaDB calculates Cosine Distance:
     distance = 1 - cosine_similarity
   - A distance of 0.0 means identical meaning.
   - A distance of 0.2 means very high relevance (similarity = 0.80).
   - A distance of 1.0 means completely unrelated.
   - In our code, we convert Distance to Similarity:
     similarity = 1.0 - distance
     This gives an intuitive percentage score (e.g. 88% match).

3. HNSW (Hierarchical Navigable Small World):
   - Finding the nearest vector by comparing against 1,000,000 items sequentially (brute force)
     would be too slow.
   - ChromaDB builds an HNSW graph (multi-layered graph like a highway network)
     to jump directly to the neighborhood of the query vector in milliseconds.
=============================================================================
"""

from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


class VectorStoreManager:
    """
    Manages embedding generation and vector search storage using ChromaDB.
    """

    def __init__(self, persist_dir: str = str(CHROMA_PERSIST_DIR)):
        self.persist_dir = persist_dir

        # 1. Load the local SentenceTransformer embedding model (100% free, runs locally)
        print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 2. Initialize ChromaDB client with persistent local storage on disk
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 3. Get or create the medical guidelines collection
        # Uses cosine distance metric for normalized embedding comparison
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of text strings into normalized 384-dimensional vector embeddings.
        """
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Ingests document chunks into ChromaDB:
        1. Extracts texts, IDs, and metadata.
        2. Computes embeddings.
        3. Upserts into ChromaDB (avoids duplicates).
        """
        if not chunks:
            return 0

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "section": chunk.get("section", "General"),
                "char_count": chunk.get("char_count", len(chunk["text"]))
            }
            for chunk in chunks
        ]

        # Generate embeddings for the chunks
        embeddings = self.embed_texts(documents)

        # Upsert: updates if ID already exists, inserts if new
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Takes a patient query, computes its embedding, and retrieves the
        top_k most semantically relevant clinical chunks with similarity scores.
        """
        if self.collection.count() == 0:
            return []

        # Embed the incoming user query using the exact same embedding model
        query_embedding = self.embed_texts([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
                # Convert Cosine Distance to Cosine Similarity score (0.0 to 1.0)
                # For normalized cosine space, similarity = 1 - distance
                similarity_score = max(0.0, min(1.0, 1.0 - dist))

                retrieved_items.append({
                    "id": doc_id,
                    "text": text,
                    "source": meta.get("source", "Unknown"),
                    "section": meta.get("section", "General"),
                    "distance": round(dist, 4),
                    "similarity": round(similarity_score, 4),
                })

        return retrieved_items

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of the indexed knowledge base.
        """
        count = self.collection.count()
        sources = set()

        if count > 0:
            all_data = self.collection.get(include=["metadatas"])
            for meta in all_data["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])

        return {
            "total_chunks": count,
            "unique_sources": sorted(list(sources)),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "collection_name": COLLECTION_NAME
        }

    def reset(self) -> None:
        """
        Clears the collection to allow fresh re-indexing.
        """
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
