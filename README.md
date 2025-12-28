# Custom-RAG-Chatbot-For-Upwork-Project

# 📄 Local RAG Chatbot: Privacy-First Knowledge Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v1.x-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)

A Proof-of-Concept (POC) for a **Retrieval-Augmented Generation (RAG)** system. This application was built as **a demo for an Upwork job**. It allows users to upload proprietary documents (PDF, DOCX) and interact with them using a locally hosted LLM, ensuring **100% data privacy** and **zero per-query costs**.

---

## 📺 Video Demo
> **Check out the demo:** You can find the recorded walk-through of the application in the `/videos` directory. It demonstrates document ingestion, vector indexing, and the grounded chat interface.

---

## 🌟 Key Features
* **Multi-Format Ingestion:** Robust processing for **PDF** and **Microsoft Word (.docx)** files.
* **Efficient Local Inference:** Powered by **Ministral-3B**—optimized for speed and low hardware requirements.
* **Privacy-First:** Designed for proprietary data; no information ever leaves the local environment.
* **Persistent Knowledge Base:** Uses **ChromaDB** to store document embeddings locally.
* **Source Grounding:** Explicitly instructed to avoid hallucinations by sticking only to provided context.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Orchestration** | LangChain (v1.x Modern Chain Syntax) |
| **LLM** | Ministral-3B (via Ollama) |
| **Embeddings** | Nomic Embed Text (8k context) |
| **Vector Store** | ChromaDB (Local Persisted) |
| **Frontend** | Streamlit |

---

## 🚀 Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.com/) and download the required models to your local machine:
```bash
ollama pull ministral-3:3b
ollama pull nomic-embed-text
```

### 2. Installation

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 🧠 How It Works



* **Ingestion:** The app uses `PyPDFLoader` and `Docx2txtLoader` to extract raw text from uploaded files.
* **Chunking:** Text is split using `RecursiveCharacterTextSplitter` into **1,000-character segments** with a 150-character overlap to preserve context across chunks.
* **Indexing:** The `nomic-embed-text` model creates high-dimensional vectors which are stored in a local `chroma_db` folder for persistent storage.
* **Retrieval:** The system uses semantic search to find the **top 6 most relevant chunks** for every user question, ensuring a broad context base.
* **Generation:** The **Ministral-3B** LLM synthesizes an answer based *only* on those retrieved chunks to ensure grounding and prevent hallucinations.

---

## 📁 Repository Structure

* **`app.py`**: The main Streamlit application logic and UI.
* **`notebooks/`**: Jupyter Notebooks documenting the RAG development, experimentation, and testing phases.
* **`videos/`**: Demonstration videos for stakeholders showing the app in a live environment.
* **`requirements.txt`**: Complete list of Python dependencies required to run the project.
