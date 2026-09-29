import os

from dotenv import load_dotenv


load_dotenv()


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:4b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

OPENSEARCH_URL = os.getenv(
    "OPENSEARCH_URL",
    "http://localhost:9200"
)

OPENSEARCH_INDEX = os.getenv(
    "OPENSEARCH_INDEX",
    "openrag_documents"
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", 1000)
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 200)
)

TOP_K = int(
    os.getenv("TOP_K", 5)
)