# YouTube Transcript RAG Engine

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

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

├── app.py          # Chat interface and RAG-based QnA pipeline <br>
├── nav.py          # Main entry point managing application navigation and configuration <br>
├── notes.py        # Automated text summarization pipeline and PDF generator <br>
└── .gitignore      # Specifies untracked files to ignore (e.g., .env, pycache)

---

## Setup and Installation

### 1. Clone the Repository

git clone [https://github.com/Sukanya-29/YouTube-Summarizer-AI.git](https://github.com/Sukanya-29/YouTube-Summarizer-AI.git) 
<br> cd YouTube-Summarizer-AI 

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Run the Application

streamlit run nav.py

---

## Final Verdict
The project architecture itself is solid—using HuggingFace embeddings locally with FAISS avoids external API embedding costs, and implementing a Map-Reduce approach for long video transcripts shows you understand context window limitations. Fix the environment variable name (`GROQ_API_KEY`), deploy it on Streamlit Cloud for a live link, and add a few badges, and this README is interview-ready.
