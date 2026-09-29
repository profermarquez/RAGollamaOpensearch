import tempfile
import time
from pathlib import Path

import streamlit as st

from rag.document_processor import DocumentProcessor
from rag.rag_service import RAGService


# =======================================================
# CONFIGURACIÓN
# =======================================================

st.set_page_config(
    page_title="OpenRAG + Ollama",
    page_icon="📚",
    layout="wide",
)


# =======================================================
# SERVICIOS
# =======================================================

@st.cache_resource
def get_processor():
    return DocumentProcessor()


@st.cache_resource
def get_rag():
    return RAGService()


processor = get_processor()
rag = get_rag()


# =======================================================
# FUNCIONES AUXILIARES
# =======================================================

def format_token(value):
    if value is None:
        return None

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return None


def format_seconds(value):
    if value is None:
        return None

    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return None


def show_metrics(metrics):
    """
    Muestra únicamente métricas realmente disponibles.
    No inventa ceros ni muestra N/D innecesarios.
    """

    if not metrics:
        return

    input_tokens = format_token(
        metrics.get("input_tokens")
    )

    output_tokens = format_token(
        metrics.get("output_tokens")
    )

    total_tokens = format_token(
        metrics.get("total_tokens")
    )

    retrieval_time = format_seconds(
        metrics.get("retrieval_time")
    )

    generation_time = format_seconds(
        metrics.get("generation_time")
    )

    response_time = format_seconds(
        metrics.get("response_time")
    )

    wall_time = format_seconds(
        metrics.get("wall_time")
    )

    # ---------------------------------------------------
    # TOKENS
    # ---------------------------------------------------

    token_parts = []

    if input_tokens is not None:
        token_parts.append(
            f"Entrada: {input_tokens}"
        )

    if output_tokens is not None:
        token_parts.append(
            f"Salida: {output_tokens}"
        )

    if total_tokens is not None:
        token_parts.append(
            f"Total: {total_tokens}"
        )

    if token_parts:
        st.caption(
            "📊 "
            + " tokens · ".join(token_parts)
            + " tokens"
        )

    # ---------------------------------------------------
    # TIEMPOS
    # ---------------------------------------------------

    time_parts = []

    if retrieval_time is not None:
        time_parts.append(
            f"🔎 Recuperación: {retrieval_time}"
        )

    if generation_time is not None:
        time_parts.append(
            f"🤖 Generación: {generation_time}"
        )

    if response_time is not None:
        time_parts.append(
            f"⚙️ RAG: {response_time}"
        )

    if wall_time is not None:
        time_parts.append(
            f"⏱️ Total: {wall_time}"
        )

    if time_parts:
        st.caption(
            " · ".join(time_parts)
        )


def show_sources(
    sources,
    title="Fuentes utilizadas"
):
    if not sources:
        return

    with st.expander(title):

        for source in sources:

            filename = source.get(
                "source",
                "desconocido"
            )

            chunk = source.get(
                "chunk"
            )

            score = source.get(
                "score"
            )

            st.markdown(
                f"📄 **{filename}**"
            )

            details = []

            if chunk is not None:
                details.append(
                    f"Fragmento: {chunk}"
                )

            if score is not None:

                try:
                    details.append(
                        f"Score: {float(score):.4f}"
                    )
                except (TypeError, ValueError):
                    pass

            if details:
                st.caption(
                    " · ".join(details)
                )


# =======================================================
# SESSION STATE
# =======================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# =======================================================
# SIDEBAR
# =======================================================

with st.sidebar:

    st.title("📚 Documentos")

    uploaded_files = st.file_uploader(
        "Seleccioná uno o varios PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.write(
            f"{len(uploaded_files)} archivo(s) seleccionado(s)"
        )

        for uploaded_file in uploaded_files:
            st.caption(
                f"📄 {uploaded_file.name}"
            )

    process_button = st.button(
        "Procesar documentos",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.subheader(
        "Documentos indexados"
    )

    if not st.session_state.indexed_files:

        st.caption(
            "Todavía no se cargaron documentos."
        )

    else:

        for filename in st.session_state.indexed_files:

            st.write(
                f"✅ {filename}"
            )


# =======================================================
# PROCESAMIENTO DE DOCUMENTOS
# =======================================================

if process_button:

    if not uploaded_files:

        st.warning(
            "Seleccioná al menos un PDF."
        )

    else:

        total_chunks = 0
        processed_count = 0
        skipped_count = 0

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            # ------------------------------------------------
            # Evitar reprocesar
            # ------------------------------------------------

            if (
                uploaded_file.name
                in st.session_state.indexed_files
            ):

                skipped_count += 1

                status.info(
                    f"{uploaded_file.name} ya estaba "
                    f"indexado. Se omite."
                )

                progress.progress(
                    (index + 1)
                    / len(uploaded_files)
                )

                continue

            temp_path = None

            try:

                status.info(
                    f"Procesando {uploaded_file.name}..."
                )

                # --------------------------------------------
                # Archivo temporal
                # --------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name

                # --------------------------------------------
                # Procesamiento Docling
                # --------------------------------------------

                chunks = processor.process_pdf(
                    temp_path
                )

                if not chunks:
                    raise RuntimeError(
                        "Docling no generó fragmentos."
                    )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                for chunk_number, chunk in enumerate(
                    chunks
                ):

                    chunk.metadata["source"] = (
                        uploaded_file.name
                    )

                    chunk.metadata["chunk"] = (
                        chunk_number
                    )

                # --------------------------------------------
                # Indexación OpenSearch
                # --------------------------------------------

                rag.index_documents(
                    chunks
                )

                total_chunks += len(
                    chunks
                )

                processed_count += 1

                st.session_state.indexed_files.append(
                    uploaded_file.name
                )

            except Exception as error:

                st.error(
                    f"Error procesando "
                    f"{uploaded_file.name}: {error}"
                )

            finally:

                if temp_path:

                    Path(
                        temp_path
                    ).unlink(
                        missing_ok=True
                    )

                progress.progress(
                    (index + 1)
                    / len(uploaded_files)
                )

        # ---------------------------------------------------
        # Resultado
        # ---------------------------------------------------

        if processed_count > 0:

            message = (
                f"Procesamiento finalizado. "
                f"{processed_count} documento(s) nuevo(s), "
                f"{total_chunks} fragmentos indexados."
            )

            if skipped_count:
                message += (
                    f" {skipped_count} archivo(s) "
                    f"omitido(s) por estar ya indexados."
                )

            status.success(
                message
            )

        elif (
            uploaded_files
            and skipped_count == len(uploaded_files)
        ):

            status.info(
                "Todos los documentos seleccionados "
                "ya estaban indexados."
            )


# =======================================================
# INTERFAZ PRINCIPAL
# =======================================================

st.title(
    "🤖 OpenRAG con Ollama"
)

st.caption(
    "Docling + OpenSearch + Ollama + Streamlit"
)


# =======================================================
# ESTADO
# =======================================================

if not st.session_state.indexed_files:

    st.info(
        "Cargá y procesá al menos un documento PDF "
        "antes de comenzar el chat."
    )


# =======================================================
# HISTORIAL
# =======================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        metrics = message.get(
            "metrics"
        )

        if metrics:
            show_metrics(
                metrics
            )

        sources = message.get(
            "sources"
        )

        if sources:
            show_sources(
                sources,
                title="Fuentes"
            )


# =======================================================
# CHAT
# =======================================================

question = st.chat_input(
    "Preguntá algo sobre los documentos...",
    disabled=not bool(
        st.session_state.indexed_files
    ),
)


if question:

    # ---------------------------------------------------
    # Usuario
    # ---------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # ---------------------------------------------------
    # Asistente
    # ---------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            # --------------------------------------------
            # Tiempo total real desde Streamlit
            # --------------------------------------------

            wall_start = (
                time.perf_counter()
            )

            with st.spinner(
                "Buscando en los documentos..."
            ):

                result = rag.ask(
                    question
                )

            wall_time = (
                time.perf_counter()
                - wall_start
            )

            # --------------------------------------------
            # Respuesta
            # --------------------------------------------

            answer = result.get(
                "answer"
            )

            if not answer:
                answer = (
                    "No se obtuvo una respuesta "
                    "del modelo."
                )

            st.markdown(
                answer
            )

            # --------------------------------------------
            # Métricas
            # --------------------------------------------

            metrics = dict(
                result.get(
                    "metrics"
                )
                or {}
            )

            # Este tiempo sí es siempre real:
            metrics["wall_time"] = (
                wall_time
            )

            show_metrics(
                metrics
            )

            # --------------------------------------------
            # Fuentes
            # --------------------------------------------

            sources = result.get(
                "sources",
                []
            )

            show_sources(
                sources
            )

            # --------------------------------------------
            # Guardar historial
            # --------------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metrics": metrics,
            })

        except Exception as error:

            error_message = (
                "Se produjo un error al generar "
                f"la respuesta: {error}"
            )

            st.error(
                error_message
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })