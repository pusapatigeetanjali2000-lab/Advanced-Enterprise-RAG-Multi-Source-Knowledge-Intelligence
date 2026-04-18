# Advanced-Enterprise-RAG-Multi-Source-Knowledge-Intelligence
📂 Advanced Enterprise RAG: Multi-Source Knowledge Intelligence

Executive Summary
This project is a production-grade Retrieval-Augmented Generation (RAG) system designed to bridge the gap between static enterprise data and actionable AI intelligence. It allows users to query unstructured documents (PDFs, Text, Docs) through a cinematic, high-authority interface while maintaining strict data privacy through local vector storage.


🚀 Key Technical Features
Multi-Source Data Ingestion: Automated pipeline using SimpleDirectoryReader to process diverse file formats.

Persistent Vector Storage: Integrated with ChromaDB to ensure that document embeddings are stored locally and are persistent across system restarts.

Context-Aware Retrieval: Implements semantic search to find the most relevant document "chunks" before generating an answer.

Source Citation (Anti-Hallucination): Includes a specialized UI component that displays the exact source snippets used by the LLM to verify accuracy.

LLM Agnostic Architecture: Built to switch seamlessly between OpenAI and Google Gemini models.

🛠️ The Tech Stack
Orchestration: LlamaIndex

Vector Database: ChromaDB

LLMs: OpenAI GPT-3.5-Turbo / Google Gemini 1.5 Flash

Frontend: Streamlit (Custom Dark-Mode CSS)

Environment: Python 3.12 + PowerShell Automation

📐 How It Works
Ingestion: Files are placed in the ./data folder.

Vectorization: The system converts text into high-dimensional embeddings.

Indexing: Vectors are stored in the persistent ChromaDB "brain."

Querying: When a user asks a question, the system performs a similarity search.

Response: The LLM synthesizes the final answer using only the retrieved facts.
Project Structure:

├── .venv/               # Virtual environment
├── chroma_db/           # Persistent vector storage
├── data/                # Knowledge base (Place your PDFs here)
├── app.py               # Main Streamlit application
├── .env                 # API Keys (Excluded via .gitignore)
├── .gitignore           # Security rules for GitHub
└── requirements.txt     # Project dependencies

pip install -r requirements.txt
