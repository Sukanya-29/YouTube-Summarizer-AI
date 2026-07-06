import streamlit as st
import dotenv
import os

#https://www.youtube.com/watch?v=ukzFI9rgwfU

st.set_page_config(page_title="Youtube RAG Assistant", layout="wide")

dotenv.load_dotenv()
groq_api = os.getenv("groq_api")


pg1 = st.Page("app.py", title="QnA")
pg2 = st.Page("notes.py", title="Note Generation")

with st.sidebar:
    st.header("Settings")
    
    # 1. API Key Logic
    if not groq_api:
        # If not in .env, ask user and save to session_state
        st.session_state.groq_api = st.text_input(
            "Enter Groq API Key", 
            value=st.session_state.get("groq_api", ""),
            type="password"
        )
    else:
        st.session_state.groq_api = groq_api
        st.success("API Key loaded from environment")

    st.session_state.url = st.text_input(
        "Paste YouTube Video URL:", 
        value=st.session_state.get("url", ""),
        placeholder="https://www.youtube.com/watch?v=..."
    )


pg = st.navigation([pg1, pg2])
pg.run()

