"""
retriever.py
=============================================================================
Clinical Retrieval Engine with Semantic Search, Entity Splitting & Deduplication.

Core Concepts to Master:
1. The Problem with Large Query Reports:
   - If a patient uploads an entire lab report with 15 tests, passing the entire
     page as a single embedding query causes "vector blur" (the vector is an average
     of all 15 tests).
   - Solution: Our ClinicalRetriever extracts key clinical terms or abnormal flags,
     queries the vector database for both the holistic text and specific flagged markers,
     and merges the results.

2. Score-Based Threshold Filtering:
   - In healthcare, returning an irrelevant chunk is worse than returning nothing,
     because it confuses the model.
   - We filter out chunks below a minimum cosine similarity threshold (e.g., 0.35).

3. Context Formatting for LLM Grounding:
   - We format retrieved chunks into numbered blocks:
     [Source 1 | metabolic_panel_cmp.md | Section: Kidney Function]
     "Creatinine reference range: 0.6 - 1.3 mg/dL..."
   - This gives the LLM explicit anchors to cite in its answer.
=============================================================================
"""

import re
from typing import List, Dict, Any, Set
from src.vector_store import VectorStoreManager


# Common clinical markers to scan for in patient reports for targeted retrieval
KNOWN_CLINICAL_MARKERS = [
    "glucose", "hba1c", "hemoglobin a1c", "blood sugar",
    "creatinine", "egfr", "bun", "blood urea nitrogen", "kidney",
    "alt", "ast", "sgpt", "sgot", "bilirubin", "alkaline phosphatase", "alp", "liver",
    "cholesterol", "ldl", "hdl", "triglycerides", "lipid",
    "wbc", "white blood cell", "rbc", "red blood cell", "hemoglobin", "hematocrit", "platelets",
    "tsh", "thyroid", "free t4", "free t3",
    "sodium", "potassium", "calcium", "electrolytes",
    "stent", "aspirin", "plavix", "clopidogrel", "discharge", "chest pain", "fever", "wound"
]


class ClinicalRetriever:
    """
    Coordinates semantic retrieval, entity-guided search, and context assembly.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def extract_clinical_keywords(self, text: str) -> List[str]:
        """
        Scans a patient report or question for high-priority clinical biomarkers.
        """
        text_lower = text.lower()
        found_markers = []

        for marker in KNOWN_CLINICAL_MARKERS:
            # Word boundary search to avoid matching inside other words
            if re.search(r'\b' + re.escape(marker) + r'\b', text_lower):
                found_markers.append(marker)

        return found_markers[:6]  # Limit to top 6 markers to keep queries focused

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant clinical guideline chunks for a query or report.
        Performs multi-query search if multiple clinical entities are detected.
        """
        # 1. Primary holistic query
        results = self.vector_store.similarity_search(query=query, top_k=top_k)

        # 2. Targeted sub-queries for specific detected markers
        keywords = self.extract_clinical_keywords(query)
        seen_ids: Set[str] = {r["id"] for r in results}

        for kw in keywords:
            kw_results = self.vector_store.similarity_search(query=f"Clinical guidelines for {kw}", top_k=2)
            for r in kw_results:
                if r["id"] not in seen_ids:
                    results.append(r)
                    seen_ids.add(r["id"])

        # 3. Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r["similarity"] >= similarity_threshold
        ]

        # 4. Sort by highest similarity first and take top_k (with up to 2 extra if high relevance)
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
        max_allowed = min(len(filtered_results), top_k + 2)
        final_chunks = filtered_results[:max_allowed]

        return final_chunks

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Transforms retrieved chunks into an unambiguous, structured text block
        for the LLM prompt.
        """
        if not chunks:
            return "NO RELEVANT CLINICAL GUIDELINES FOUND IN THE KNOWLEDGE BASE."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"[Source {idx}: {chunk['source']} | Section: {chunk['section']} | Match: {int(chunk['similarity']*100)}%]\n"
                f"{chunk['text']}\n"
            )
            context_blocks.append(block)

        return "\n----------------------------------------\n".join(context_blocks)
