# Entha: Multi-Hop Retriever for Context-Enriched RAG

**Entha** is a proof-of-concept (PoC) implementation of  **multi-hop document retrieval** that enhances traditional Retrieval-Augmented Generation (RAG) pipelines. It improves contextual understanding by not only retrieving the top-*k* most relevant documents for a query, but also expanding context by retrieving top-*n* semantically similar documents based on those initial results.

---

## 🔍 Motivation

Traditional RAG systems retrieve a fixed number of documents (top-*k*) based solely on query relevance. However, this can miss related but indirectly connected context. **Entha** addresses this by:

1. Retrieving the top-*k* documents relevant to the user's query.
2. For each of those documents, retrieving top-*n* additional documents based on embedding similarity.
3. Combining all results to create a richer and more connected context for the LLM to generate responses.

---

## 🚀 Features

- ✅ Multi-hop context retrieval (top-*k* + neighbors of top-*k*)
- ✅ Google Generative AI Embeddings for semantic similarity
- ✅ ChromaDB for vector storage and retrieval
- ✅ Streamlit-based interface for easy interaction
- ✅ Modular design for plug-and-play LLMs

---

## 📦 Tech Stack

- Python
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [Google Generative AI](https://ai.google.dev/)
- Streamlit

---

## 🛠 How It Works

```text
User Query
   │
   ├──▶ Top-K Document Retrieval (based on embeddings)
   │         │
   │         └──▶ Top-N Neighbors (per document)
   │                     │
   └────────────────────▶ Combined Context
                          ↓
              Sent to LLM (Gemini)
