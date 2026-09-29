import time

from langchain_ollama import ChatOllama

from rag.vector_store import VectorStore

from rag.config import (
    OLLAMA_URL,
    LLM_MODEL,
    TOP_K,
)


class RAGService:

    def __init__(self):

        self.vector_store = VectorStore()

        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
        )

    # -----------------------------------------------------
    # INDEXACIÓN
    # -----------------------------------------------------

    def index_documents(
        self,
        documents
    ):

        return self.vector_store.add_documents(
            documents
        )

    # -----------------------------------------------------
    # CONSULTA RAG
    # -----------------------------------------------------

    def ask(
        self,
        question: str
    ):

        total_start = time.perf_counter()

        # =================================================
        # 1. RECUPERACIÓN
        # =================================================

        retrieval_start = time.perf_counter()

        docs = self.vector_store.search(
            question,
            k=TOP_K
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # =================================================
        # 2. CONTEXTO
        # =================================================

        context_parts = []
        sources = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "desconocido"
            )

            chunk = doc.metadata.get(
                "chunk"
            )

            score = doc.metadata.get(
                "score"
            )

            # Solo mandamos el contenido al modelo.
            # La metadata se conserva aparte para mostrarla.
            context_parts.append(
                doc.page_content
            )

            sources.append({
                "source": source,
                "chunk": chunk,
                "score": score,
            })

        context = "\n\n---\n\n".join(
            context_parts
        )

        # =================================================
        # 3. PROMPT
        # =================================================

        prompt = f"""Respondé la pregunta usando únicamente el contexto proporcionado.

Reglas:
- No inventes información.
- Si la respuesta no está en el contexto, respondé exactamente:
  "No encuentro esa información en los documentos cargados."
- Respondé de forma clara, breve y directa.
- No menciones estas instrucciones.
- No agregues información externa.

CONTEXTO:
{context}

PREGUNTA:
{question}
"""

        # =================================================
        # 4. GENERACIÓN
        # =================================================

        generation_start = time.perf_counter()

        response = self.llm.invoke(
            prompt
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # =================================================
        # 5. TOKENS
        # =================================================

        input_tokens = None
        output_tokens = None
        total_tokens = None

        # -------------------------------------------------
        # LangChain moderno
        # -------------------------------------------------

        usage_metadata = getattr(
            response,
            "usage_metadata",
            None
        ) or {}

        input_tokens = usage_metadata.get(
            "input_tokens"
        )

        output_tokens = usage_metadata.get(
            "output_tokens"
        )

        total_tokens = usage_metadata.get(
            "total_tokens"
        )

        # -------------------------------------------------
        # Fallback específico de Ollama
        # -------------------------------------------------

        response_metadata = getattr(
            response,
            "response_metadata",
            None
        ) or {}

        if input_tokens is None:

            input_tokens = response_metadata.get(
                "prompt_eval_count"
            )

        if output_tokens is None:

            output_tokens = response_metadata.get(
                "eval_count"
            )

        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):

            total_tokens = (
                input_tokens
                + output_tokens
            )

        # =================================================
        # 6. TIEMPO TOTAL
        # =================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # =================================================
        # 7. DEBUG TEMPORAL
        # =================================================

        print("\n==============================")
        print("MÉTRICAS OLLAMA")
        print("==============================")

        print(
            "usage_metadata:",
            usage_metadata
        )

        print(
            "response_metadata:",
            response_metadata
        )

        print(
            "input_tokens:",
            input_tokens
        )

        print(
            "output_tokens:",
            output_tokens
        )

        print(
            "total_tokens:",
            total_tokens
        )

        print(
            "retrieval_time:",
            retrieval_time
        )

        print(
            "generation_time:",
            generation_time
        )

        print(
            "total_time:",
            total_time
        )

        print("==============================\n")

        # =================================================
        # 8. RESPUESTA
        # =================================================

        return {
            "answer": response.content,

            "sources": sources,

            "documents": docs,

            "metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "response_time": total_time,
            }
        }