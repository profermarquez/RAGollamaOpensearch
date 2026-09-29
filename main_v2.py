from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


PDF_PATH = "propuesta.pdf"


# ---------------------------------------------------------
# 1. Cargar PDF
# ---------------------------------------------------------

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Paginas cargadas: {len(documents)}")


# ---------------------------------------------------------
# 2. Dividir documento
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks generados: {len(chunks)}")


# ---------------------------------------------------------
# 3. Modelo de embeddings Ollama
# ---------------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ---------------------------------------------------------
# 4. Crear base vectorial
# ---------------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="documentos"
)


# ---------------------------------------------------------
# 5. Crear modelo LLM
# ---------------------------------------------------------

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# ---------------------------------------------------------
# 6. Pregunta
# ---------------------------------------------------------

question = input("\nPregunta: ")


# ---------------------------------------------------------
# 7. Recuperar documentos relevantes
# ---------------------------------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

docs = retriever.invoke(question)


# ---------------------------------------------------------
# 8. Construir contexto
# ---------------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# ---------------------------------------------------------
# 9. Prompt
# ---------------------------------------------------------

prompt = f"""
Sos un asistente que responde únicamente utilizando
la información incluida en el contexto.

Si la respuesta no se encuentra en el contexto,
respondé:

"No encuentro esa información en los documentos."

CONTEXTO:

{context}

PREGUNTA:

{question}
"""


# ---------------------------------------------------------
# 10. Generar respuesta
# ---------------------------------------------------------

response = llm.invoke(prompt)


print("\nRESPUESTA:")
print(response.content)