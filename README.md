# RAG System with Hugging Face & LangChain

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline. It is designed to ingest documentation, index it into a vector database, and provide precise answers to user queries using state-of-the-art open-source models.

---

## 📂 Project Structure

The project is organized into modular components for better maintainability:

* **`main.py`**: The entry point. Orchestrates the loading, indexing, and querying process.
* **`config.py`**: Centralized configuration for model names, hyperparameters, and text-splitting rules.
* **`data_loader.py`**: Logic for fetching and parsing the knowledge base (Hugging Face Documentation).
* **`retriever.py`**: Handles text splitting (hierarchical), embeddings generation, and FAISS vector store management.
* **`reader.py`**: Sets up the LLM (Zephyr-7B) with 4-bit quantization and manages prompt templates.
* **`requirements.txt`**: List of all Python dependencies.
* **`evaluator.py`**: Evaluates the RAG system using Ragas metrics.

---

## 🚀 Features

* **Hierarchical Markdown Splitting**: Intelligent chunking that respects headers and code blocks.
* **Semantic Search**: Uses the `GTE-small` embedding model for high-accuracy retrieval.
* **Memory-Efficient LLM**: Implements `bitsandbytes` 4-bit quantization to run a 7B parameter model on limited VRAM.
* **Vector Indexing**: Powered by `FAISS` for fast similarity searches.
* **Evaluation**: Uses `Ragas` to evaluate the RAG system.
