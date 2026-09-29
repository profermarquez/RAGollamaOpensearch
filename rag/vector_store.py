from opensearchpy import OpenSearch, helpers

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from rag.config import (
    OLLAMA_URL,
    EMBEDDING_MODEL,
    OPENSEARCH_URL,
    OPENSEARCH_INDEX,
)


class VectorStore:

    def __init__(self):

        # -------------------------------------------------
        # EMBEDDINGS OLLAMA
        # -------------------------------------------------

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL,
        )

        # -------------------------------------------------
        # CLIENTE OPENSEARCH
        # -------------------------------------------------

        self.client = OpenSearch(
            hosts=[OPENSEARCH_URL],
            use_ssl=False,
            verify_certs=False,
        )

        # -------------------------------------------------
        # Obtener dimensión real del modelo
        # -------------------------------------------------

        test_vector = self.embeddings.embed_query(
            "test"
        )

        self.dimension = len(test_vector)

        print(
            f"[OpenSearch] Dimensión embeddings: {self.dimension}"
        )

        # -------------------------------------------------
        # Crear índice
        # -------------------------------------------------

        self._create_index_if_not_exists()

    # =====================================================
    # CREAR ÍNDICE
    # =====================================================

    def _create_index_if_not_exists(self):

        if self.client.indices.exists(
            index=OPENSEARCH_INDEX
        ):

            print(
                f"[OpenSearch] Índice existente: {OPENSEARCH_INDEX}"
            )

            return

        mapping = {

            "settings": {

                "index": {
                    "knn": True
                }

            },

            "mappings": {

                "properties": {

                    "embedding": {

                        "type": "knn_vector",

                        "dimension": self.dimension,

                        "method": {

                            "name": "hnsw",

                            # IMPORTANTE:
                            # OpenSearch 3.x
                            "engine": "lucene",

                            "space_type": "cosinesimil",

                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },

                    "text": {
                        "type": "text"
                    },

                    "metadata": {
                        "type": "object"
                    },

                    "source": {
                        "type": "keyword"
                    },

                    "chunk": {
                        "type": "integer"
                    }
                }
            }
        }

        self.client.indices.create(
            index=OPENSEARCH_INDEX,
            body=mapping
        )

        print(
            f"[OpenSearch] Índice creado: {OPENSEARCH_INDEX}"
        )

    # =====================================================
    # AGREGAR DOCUMENTOS
    # =====================================================

    def add_documents(
        self,
        documents
    ):

        if not documents:
            return []

        # -------------------------------------------------
        # Obtener textos
        # -------------------------------------------------

        texts = [
            document.page_content
            for document in documents
        ]

        print(
            f"[OpenSearch] Generando embeddings "
            f"para {len(texts)} fragmentos..."
        )

        # -------------------------------------------------
        # Generar embeddings en lote
        # -------------------------------------------------

        vectors = self.embeddings.embed_documents(
            texts
        )

        # -------------------------------------------------
        # Crear operaciones bulk
        # -------------------------------------------------

        actions = []

        for document, vector in zip(
            documents,
            vectors
        ):

            metadata = (
                document.metadata
                if document.metadata
                else {}
            )

            source = metadata.get(
                "source",
                "desconocido"
            )

            chunk = metadata.get(
                "chunk",
                0
            )

            action = {

                "_index": OPENSEARCH_INDEX,

                "_source": {

                    "text": document.page_content,

                    "embedding": vector,

                    "metadata": metadata,

                    "source": source,

                    "chunk": chunk,
                }
            }

            actions.append(
                action
            )

        # -------------------------------------------------
        # Bulk insert
        # -------------------------------------------------

        success, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
        )

        # -------------------------------------------------
        # Refrescar índice
        # -------------------------------------------------

        self.client.indices.refresh(
            index=OPENSEARCH_INDEX
        )

        print(
            f"[OpenSearch] Fragmentos indexados: {success}"
        )

        if errors:

            print(
                "[OpenSearch] Errores de indexación:"
            )

            for error in errors:
                print(error)

        return success

    # =====================================================
    # BÚSQUEDA VECTORIAL
    # =====================================================

    def search(
        self,
        query: str,
        k: int = 5
    ):

        # -------------------------------------------------
        # Embedding de la pregunta
        # -------------------------------------------------

        query_vector = self.embeddings.embed_query(
            query
        )

        # -------------------------------------------------
        # Query KNN
        # -------------------------------------------------

        search_body = {

            "size": k,

            "query": {

                "knn": {

                    "embedding": {

                        "vector": query_vector,

                        "k": k
                    }
                }
            }
        }

        response = self.client.search(
            index=OPENSEARCH_INDEX,
            body=search_body
        )

        # -------------------------------------------------
        # Convertir resultados a Document
        # -------------------------------------------------

        documents = []

        for hit in response["hits"]["hits"]:

            data = hit["_source"]

            metadata = data.get(
                "metadata",
                {}
            )

            # Score de similitud
            metadata["score"] = hit.get(
                "_score",
                0
            )

            document = Document(
                page_content=data.get(
                    "text",
                    ""
                ),
                metadata=metadata
            )

            documents.append(
                document
            )

        return documents