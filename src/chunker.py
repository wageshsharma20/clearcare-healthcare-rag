"""
chunker.py
=============================================================================
Document Ingestion & Text Chunking Engine.

Core Concepts to Master:
1. Why Chunking is Mandatory:
   - Raw documents (like 50-page clinical guidelines) cannot be embedded as one
     giant vector because embeddings represent the *average* meaning of the text.
     If a document contains 50 different diseases, its single vector is diluted
     and useless for pinpoint queries.
   - By dividing text into small, concept-specific chunks (e.g., just the
     Creatinine explanation), the vector for that chunk has a sharp, focused
     semantic representation.

2. Recursive Semantic Splitting:
   - Instead of blindly cutting every N characters (which might cut "not safe"
     into "not" and "safe"), we try to split at natural human boundaries:
     Markdown Headers (##) -> Double Line Breaks (\n\n) -> Sentences (. ) -> Spaces.

3. Chunk Overlap:
   - Preserves continuity so that context isn't lost if a medical exception or
     dosage warning straddles two adjacent chunks.

4. Metadata Enrichment:
   - We attach source file names, section titles, and chunk indices to every
     chunk so we can provide exact citations back to the user.
=============================================================================
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any


class MedicalChunker:
    """
    Splits clinical reference documents into structured chunks with rich metadata.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source_name: str) -> List[Dict[str, Any]]:
        """
        Recursively splits text into overlapping chunks, tracking the active
        section header for each chunk to maintain clinical context.
        """
        chunks: List[Dict[str, Any]] = []

        # Split document by paragraphs/sections first
        # We also detect markdown headers like '# ', '## ', '### ' to track section context
        lines = text.split("\n")
        current_section = "General Overview"
        current_buffer = ""
        chunk_counter = 0

        for line in lines:
            stripped = line.strip()

            # If line is a markdown header, update current section title
            if stripped.startswith("#"):
                current_section = stripped.lstrip("#").strip()

            # Check if adding this line exceeds target chunk size
            if len(current_buffer) + len(line) + 1 > self.chunk_size and len(current_buffer) > 0:
                # Save current buffer as a chunk
                chunk_id = f"{Path(source_name).stem}_chunk_{chunk_counter}"
                chunks.append({
                    "id": chunk_id,
                    "text": current_buffer.strip(),
                    "source": Path(source_name).name,
                    "section": current_section,
                    "char_count": len(current_buffer.strip()),
                })
                chunk_counter += 1

                # Carry over the last `chunk_overlap` characters for continuity
                if len(current_buffer) > self.chunk_overlap:
                    current_buffer = current_buffer[-self.chunk_overlap:] + "\n" + line
                else:
                    current_buffer = line
            else:
                if current_buffer:
                    current_buffer += "\n" + line
                else:
                    current_buffer = line

        # Don't forget the last remaining buffer
        if current_buffer.strip():
            chunk_id = f"{Path(source_name).stem}_chunk_{chunk_counter}"
            chunks.append({
                "id": chunk_id,
                "text": current_buffer.strip(),
                "source": Path(source_name).name,
                "section": current_section,
                "char_count": len(current_buffer.strip()),
            })

        return chunks

    def load_and_chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Loads a file (.md, .txt, or .pdf) and returns its chunks.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = ""
        suffix = file_path.suffix.lower()

        if suffix in [".md", ".txt"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        elif suffix == ".pdf":
            # Extract text page-by-page from PDF
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += f"\n--- Page {page_idx + 1} ---\n" + page_text
            except Exception as e:
                raise RuntimeError(f"Error reading PDF {file_path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: .md, .txt, .pdf")

        return self.split_text(text, source_name=file_path.name)

    def load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """
        Loads and chunks all markdown and text files within a directory.
        """
        all_chunks: List[Dict[str, Any]] = []
        dir_path = Path(dir_path)

        for file_path in dir_path.glob("**/*"):
            if file_path.suffix.lower() in [".md", ".txt", ".pdf"]:
                chunks = self.load_and_chunk_file(file_path)
                all_chunks.extend(chunks)

        return all_chunks
