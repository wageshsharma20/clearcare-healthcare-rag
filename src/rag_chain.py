"""
rag_chain.py
=============================================================================
Augmented Generation Pipeline & Medical Prompt Orchestration Engine.

Core Concepts to Master:
1. Grounded Generation (Zero-Shot Medical Persona):
   - We give the LLM a specialized system persona: An empathetic clinical educator
     who translates technical medical findings into accessible language.
   - We strictly ban speculative diagnoses ("You have kidney failure") in favor of
     educational interpretations ("Your eGFR is 48, which indicates moderate reduction
     in filtration; you should review hydration and medications with your doctor").

2. Two Special Workflows:
   - A. `translate_report(report_text)`: Comprehensive breakdown of an entire lab or
     discharge document (Executive summary, Biomarker traffic light, Doctor questions).
   - B. `answer_question(question, chat_history)`: Direct conversational Q&A for
     patient questions with source citations.

3. Groq API Adapter with Safe Fallback:
   - Uses Groq's high-speed inference engine (`llama-3.3-70b-versatile`).
   - If an API key is not yet set, provides an informative local mock generation
     highlighting the exact retrieved guidelines so you can test immediately.
=============================================================================
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq

from src.config import DEFAULT_GROQ_MODEL, GROQ_API_KEY
from src.retriever import ClinicalRetriever


# Clinical System Prompt for Full Lab Report Translation
REPORT_TRANSLATION_SYSTEM_PROMPT = """You are ClearCare AI, an empathetic and highly knowledgeable healthcare communication assistant.
Your goal is to help patients understand their medical lab results and hospital discharge notes in plain, comforting, 8th-grade level English.

CRITICAL CLINICAL SAFETY RULES:
1. Grounding: Rely strictly on the PROVIDED CLINICAL GUIDELINES below. Cite references using [Source X] format.
2. No Definite Diagnoses: Never say "You have X disease." Say "This result often indicates..." or "Doctors typically check this to evaluate..."
3. Categorization: Organize test results into three clear traffic-light groups:
   - 🟢 NORMAL / HEALTHY RANGE
   - 🟡 BORDERLINE / MILD VARIATION (Commonly influenced by diet, hydration, exercise)
   - 🔴 NEEDS ATTENTION / OUT OF RANGE (Should be discussed promptly with your physician)
4. Context & Reassurance: Explain benign causes (e.g., mild dehydration raising BUN, vigorous exercise raising ALT) to avoid unnecessary patient panic.
5. Action Plan: End with a bulleted "Questions to Ask Your Doctor" checklist.
6. Mandatory Disclaimer: Always include a prominent medical disclaimer at the bottom.
"""

# Clinical System Prompt for Conversational Q&A
CHAT_SYSTEM_PROMPT = """You are ClearCare AI, a clinical patient education assistant.
Answer the patient's question clearly, warmly, and accurately using ONLY the provided clinical excerpts.

SAFETY RULES:
- Use simple everyday language (avoid medical jargon or define it immediately).
- Reference the clinical guidelines using [Source X] citations.
- If the guidelines do not contain the answer, honestly state: "I don't have enough verified clinical information in my current guidelines to answer this safely. Please consult your physician."
- Never recommend stopping prescribed medications (especially heart medications or antibiotics).
"""


class ClearCareRAGChain:
    """
    Orchestrates the retrieval and generation phases using Groq's LLM.
    """

    def __init__(self, retriever: ClinicalRetriever, api_key: Optional[str] = None):
        self.retriever = retriever
        self.api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.model_name = DEFAULT_GROQ_MODEL
        self._init_groq_client()

    def set_api_key(self, api_key: str):
        """Allows setting or updating the Groq API key at runtime (e.g. from Streamlit UI)."""
        self.api_key = api_key.strip()
        self._init_groq_client()

    def _init_groq_client(self):
        """Initializes the Groq client if an API key is available."""
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                self.client = None
        else:
            self.client = None

    def translate_report(
        self,
        report_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Executes full RAG workflow on a patient report or discharge summary:
        1. Retrieves relevant clinical knowledge base chunks.
        2. Formats augmented prompt.
        3. Calls Groq LLM to generate plain-English translation.
        """
        # Step 1: Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=report_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        formatted_context = self.retriever.format_context_for_prompt(retrieved_chunks)

        user_prompt = f"""
PATIENT REPORT TO TRANSLATE:
\"\"\"
{report_text}
\"\"\"

VERIFIED CLINICAL GUIDELINES FROM KNOWLEDGE BASE:
\"\"\"
{formatted_context}
\"\"\"

Please provide a compassionate, structured translation following the system rules:
1. Executive Summary (What this report means overall in 2-3 simple sentences).
2. Traffic-Light Findings Table / Breakdown (🟢 Normal, 🟡 Borderline, 🔴 Needs Discussion).
3. Plain-English Explanations with [Source X] citations.
4. "Questions to Ask Your Doctor" Checklist.
5. Medical Disclaimer.
"""

        # Step 2: Generation
        if self.client:
            try:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": REPORT_TRANSLATION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                        max_tokens=2048,
                    )
                except Exception as model_err:
                    if "does not exist" in str(model_err) or "access" in str(model_err):
                        response = self.client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": REPORT_TRANSLATION_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.2,
                            max_tokens=2048,
                        )
                    else:
                        raise model_err
                generated_text = response.choices[0].message.content
            except Exception as e:
                generated_text = f"⚠️ Groq API Error: {str(e)}\nPlease verify your Groq API Key."
        else:
            # Fallback mock mode when no API key is provided
            generated_text = self._generate_mock_translation(report_text, retrieved_chunks)

        return {
            "response": generated_text,
            "retrieved_chunks": retrieved_chunks,
            "raw_prompt": user_prompt
        }

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """
        Conversational Q&A grounded in retrieved clinical guidelines.
        """
        # Step 1: Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        formatted_context = self.retriever.format_context_for_prompt(retrieved_chunks)

        user_prompt = f"""
PATIENT QUESTION:
{question}

VERIFIED CLINICAL GUIDELINES:
{formatted_context}

Please provide an answer in plain, reassuring language with citations to the sources above.
"""

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        # Append previous conversation history if provided
        if chat_history:
            for msg in chat_history[-4:]:  # Keep last 2-3 turns for context
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        # Step 2: Generation
        if self.client:
            try:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                except Exception as model_err:
                    if "does not exist" in str(model_err) or "access" in str(model_err):
                        response = self.client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=messages,
                            temperature=0.2,
                            max_tokens=1024,
                        )
                    else:
                        raise model_err
                generated_text = response.choices[0].message.content
            except Exception as e:
                generated_text = f"⚠️ Groq API Error: {str(e)}\nPlease verify your Groq API Key."
        else:
            generated_text = (
                f"### [Offline Preview - No Groq API Key Set]\n\n"
                f"We retrieved **{len(retrieved_chunks)} clinical guidelines** matching your question:\n\n"
            )
            for idx, c in enumerate(retrieved_chunks, 1):
                generated_text += f"- **[Source {idx}: {c['source']}]** ({c['section']}, Match: {int(c['similarity']*100)}%): *\"{c['text'][:140]}...\"*\n\n"
            generated_text += (
                "\n💡 *To see full AI-generated answers, please enter a free Groq API key in the sidebar.*"
            )

        return {
            "response": generated_text,
            "retrieved_chunks": retrieved_chunks,
            "raw_prompt": user_prompt
        }

    def _generate_mock_translation(self, report_text: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an educational preview when running without an active API key.
        """
        return f"""### 🩺 ClearCare Clinical Translation (Offline Preview Mode)

> **Note:** To enable live LLM generation, please paste your free Groq API Key in the left sidebar!

#### 1. Executive Summary
Your document was successfully analyzed by the **ClearCare RAG Pipeline**. The system cross-referenced your report against our verified medical guidelines, identifying **{len(chunks)} relevant clinical sections**.

#### 2. Retrieved Clinical Grounding Evidence
The following clinical references were retrieved by ChromaDB with high semantic similarity:
""" + "\n".join([
    f"- **[Source {i+1}: {c['source']}]** (Section: *{c['section']}*, Cosine Match: **{int(c['similarity']*100)}%**)\n"
    f"  > \"{c['text'][:200]}...\""
    for i, c in enumerate(chunks)
]) + """

#### 3. Recommended Questions to Ask Your Doctor
1. *"Could you please explain what these results mean in the context of my overall health and lifestyle?"*
2. *"Are there any repeat lab tests or imaging studies we should schedule in 3 to 6 months?"*
3. *"Do any of my current daily medications or supplements need adjustment based on these numbers?"*

---
*Disclaimer: ClearCare AI is an educational communication tool and does not provide formal medical diagnosis or prescription advice. Always discuss results with a licensed physician.*
"""
