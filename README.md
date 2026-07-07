# YouTube RAG Assistant

A multi-page Streamlit application designed to extract insights, perform Question-Answering (QnA), and automatically generate comprehensive study notes directly from YouTube video transcripts. The system leverages LangChain, Groq (Llama 3.1), and FAISS vector embeddings to deliver accurate, context-aware responses.

## Features

* **Multi-Page Architecture:** Implements Streamlit navigation to seamlessly switch between conversational QnA and automated note generation.
* **Contextual Video QnA:** Utilizes Retrieval-Augmented Generation (RAG) by chunking transcripts, computing semantic embeddings, and storing them in a local FAISS vector database for precise querying.
* **Intelligent Note Generation:** Processes long video transcripts through a Map-Reduce approach to extract dense technical concepts and format them into an executive study guide.
* **PDF Export Utility:** Converts generated summaries and guides into a downloadable PDF format for offline study.

## Technical Stack

* **Frontend:** Streamlit
* **LLM Framework:** LangChain Core, LangChain Community
* **Inference Engine:** Groq API (Model: llama-3.1-8b-instant)
* **Vector Store & Embeddings:** FAISS, HuggingFace Embeddings (sentence-transformers/all-mpnet-base-v2)
* **Data Processing:** Pytubefix, LangChain Text Splitters
* **Document Generation:** FPDF

## Project Structure

├── app.py          # Chat interface and RAG-based QnA pipeline
├── nav.py          # Main entry point managing application navigation and configuration
├── notes.py        # Automated text summarization pipeline and PDF generator
└── .gitignore      # Specifies untracked files to ignore (e.g., .env, __pycache__)

---

## Setup and Installation

### 1. Clone the Repository

git clone [https://github.com/Sukanya-29/YouTube-Summarizer-AI.git](https://github.com/Sukanya-29/YouTube-Summarizer-AI.git) 
<br> cd YouTube-Summarizer-AI 

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Configure Environment Variables

groq_api=your_actual_groq_api_key_here

### 4. Run the Application

streamlit run nav.py
