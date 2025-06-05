# RAG System: Retrieval-Augmented Generation for Intelligent Document Querying

A Retrieval-Augmented Generation (RAG) system that enables natural language querying over document collections with semantic search and AI-powered response generation.

---

## Overview

This RAG system lets users ask questions in natural language and receive contextually relevant answers by combining:

- **Vector similarity search** for document retrieval
- **Advanced re-ranking** to improve result relevance
- **Large Language Model (LLM)** for generating precise responses

---

## Architecture

- **Vector Database:** [ChromaDB](https://www.trychroma.com/) for storing and retrieving document embeddings
- **Embeddings:** `mxbai-embed-large` model via [Ollama](https://ollama.com/)
- **Language Model:** Llama 3.2 3B via Ollama
- **Re-ranking:** CrossEncoder (from sentence-transformers) to select the most relevant chunks

---

## How It Works

1. **Initialize**  
   Load vector database, embedding model, and LLM.

2. **Query**  
   User submits a natural language question.

3. **Search**  
   Find top-5 relevant document chunks using semantic similarity search.

4. **Re-rank**  
   Use CrossEncoder to select the best 2 chunks for context.

5. **Generate**  
   LLM generates the answer based on retrieved context.

6. **Return**  
   Provide the answer along with source references.

---
## Flow Chart
![image](https://github.com/user-attachments/assets/196646c0-976c-4427-a71f-99de88175495)

---
## Configuration

```python
PERSIST_DIR = "./chroma_db"
OLLAMA_MODEL_EMBED = "mxbai-embed-large:latest"
OLLAMA_MODEL_LLM = "llama3.2:3b"
```
## Demo
https://www.loom.com/share/2227e33dc47c41308baf8f082e6bb7fb?sid=52b92213-65bd-4b31-ad77-0140f4551fc8
