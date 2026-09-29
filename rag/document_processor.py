from pathlib import Path

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class DocumentProcessor:

    def __init__(self):

        self.converter = DocumentConverter()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def process_pdf(
        self,
        file_path: str
    ) -> list[Document]:

        path = Path(file_path)

        result = self.converter.convert(path)

        markdown = result.document.export_to_markdown()

        document = Document(
            page_content=markdown,
            metadata={
                "source": path.name,
                "path": str(path),
            }
        )

        chunks = self.splitter.split_documents(
            [document]
        )

        for i, chunk in enumerate(chunks):

            chunk.metadata["chunk"] = i

        return chunks