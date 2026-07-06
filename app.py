import streamlit as st

import warnings
warnings.filterwarnings("ignore") 

# from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from pytubefix import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# import dotenv
# import os 

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


groq_api = st.session_state.get("groq_api")
url = st.session_state.get("url")


@st.cache_resource(show_spinner=False)
def get_llm(api_key):
    return init_chat_model(
        model="groq:llama-3.1-8b-instant",
        api_key=api_key
    )


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def create_vector_store(url):
    try:
        yt = YouTube(url)
        caption = yt.captions.get_by_language_code('en') or yt.captions.get_by_language_code('a.en')
        
        if not caption:
            raise ValueError("No English transcript found.")
            
        transcript_text = caption.generate_srt_captions()
        docs = [Document(page_content=transcript_text, metadata={"source": url, "author": yt.author})]
        
        author = yt.author

        split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_doc = split.split_documents(docs)
        
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(split_doc, embedding=embeddings)
        return vector_store.as_retriever(), author
    
    except Exception as e:
        # Catches specific YouTube errors (Private videos, No captions)
        raise Exception(f"YouTube Error: {str(e)}")

with st.sidebar:
    # st.header("Settings")

    # if not groq_api:
    #     groq_api = st.text_input("Enter Groq API Key", type="password")
    # else:
    #     st.success("API Key loaded from environment")
    
    # url = st.text_input("Paste YouTube Video URL:", placeholder= "https://www.youtube.com/watch?v=...")
    analyze_clicked = st.button("Analyze Video")

st.title("Youtube RAG system") 
st.subheader("QnA")
st.markdown("Extract insights and ask questions about any YouTube video transcript.")

msgs = StreamlitChatMessageHistory(key="chat_messages")

chat_container = st.container()
with chat_container:
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
        

if not groq_api:
    st.warning("Please provide a Groq API key to start.")
    st.stop()

llm = get_llm(groq_api)

if "last_url" not in st.session_state:
    st.session_state.last_url = ""

if url != st.session_state.last_url:
    if "retriever" in st.session_state:
        del st.session_state.retriever
    msgs.clear()
    st.session_state.last_url = url


if analyze_clicked:
    if url:
        with st.spinner("Processing video...."):
            try:
                retriever, author = create_vector_store(url)
                st.session_state.retriever = retriever
                st.session_state.author_name = author
                st.session_state.chat_messages = [] 
                st.session_state.last_url = url
                st.success("Analysis complete!")
                st.rerun()  

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid URL first.")


if "retriever" not in st.session_state:
    st.info("Please paste a URL and click 'Analyze Video' in the sidebar to start.")
    st.chat_input("Analyze a video first...", disabled=True)
else:
    if query := st.chat_input("Ask a question about the video content:"):
        msgs.add_user_message(query)

        with chat_container:
            st.chat_message("human").write(query)

        with st.spinner("Thinking..."):
            try:
                retr_doc= st.session_state.retriever.invoke(query)
                context_text = "\n".join([doc.page_content for doc in retr_doc])

                current_author = st.session_state.get("author_name", "the speaker")
   
                prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""You are assisting with a video by {current_author}. You are a professional assistant. Answer the question using the context below. 
                        Structure your response with headings and bullet points.
                        
                        Context:
                        {context_text}"""),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])
                
                chain = prompt | llm

                chain_with_history = RunnableWithMessageHistory(
                        chain,
                        lambda session_id: msgs,
                        input_messages_key="question",
                        history_messages_key="history",
                    )

                config = {"configurable": {"session_id": "youtube_session"}}
                response = chain_with_history.invoke({"question":query}, config)
                
                st.divider()
                st.markdown("### Answer:")
                st.chat_message("ai").write(response.content)

            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")






                # prompt=f"""
                        #     You are an AI assistant. 
                        #     Your task is to answer the given question using the provided context. 
                        #     Always produce a well-structured, organized, and skimmable output. 
                        #     Use clear headings, bullet points, and tables when appropriate. 
                        #     Keep the response concise, accurate, and easy to understand. 
                        #     Do not mention the context source or meta-instructions. 
                        #     Focus only on delivering the answer in a professional and engaging format.

                        #     Context:
                        #     {context_text}

                        #     Question:
                        #     {query}

                        #     Output requirements:
                        #     - Provide a direct answer to the question.
                        #     - Organize information into sections with headings.
                        #     - Use bullet points or numbered lists for clarity.
                        #     - Include tables if comparisons are needed.
                        #     - Avoid filler phrases like "according to the context" or "based on the document".
                        #     - Keep the tone informative, clear, and engaging.
                        #     """
                 