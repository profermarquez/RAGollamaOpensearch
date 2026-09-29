# Requisitos previos en Ollama

# Modelo de lenguaje (ej. Llama 3.1 o Mistral Nemo)
ollama pull llama3.1

# Modelo de embeddings recomendado para OpenRAG
ollama pull nomic-embed-text

# Entorno virtual y dependencias
virtualenv env
/env/Scripts/activate
pip install langchain langchain-ollama langchain-community langchain-text-splitters chromadb pypdf streamlit docling
pip install langchain-chroma
pip install opensearch-py

# Ejecucion 
streamlit run main.py