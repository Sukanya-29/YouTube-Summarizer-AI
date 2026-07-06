import streamlit as st
import warnings
from pytubefix import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from fpdf import FPDF
# import dotenv
# import os

warnings.filterwarnings("ignore") 

groq_api = st.session_state.get("groq_api")
url = st.session_state.get("url")


@st.cache_resource(show_spinner=False)
def get_llm(api_key):
    return init_chat_model(
        model="groq:llama-3.1-8b-instant",
        api_key=api_key
    )

def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_text)
    return bytes(pdf.output())

with st.sidebar:
    download_placeholder = st.empty()

st.title("Youtube RAG system") 

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.subheader("Notes Generator")
with col3:
    generate_btn = st.button("GENERATE NOTES", use_container_width=True)

st.markdown("Extract insights and generate downloadable notes from YouTube transcripts.")

if not groq_api:
    st.warning("Please provide a Groq API key to start.")
    st.stop()

llm = get_llm(groq_api)

if generate_btn:
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Processing video..."):
            try:
                yt = YouTube(url)
                caption = yt.captions.get_by_language_code('en') or yt.captions.get_by_language_code('a.en')
                
                if not caption:
                    st.error("No English transcript available for this video.")
                    st.stop()
                
                full_transcript = caption.generate_srt_captions()
                video_title = yt.title

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
                chunks = text_splitter.split_text(full_transcript)
                
                partial_summaries = []
                progress_bar = st.progress(0)

                for i, chunk in enumerate(chunks):
                    chunk_prompt = f"""
                        Act as a technical scribe. Extract core concepts from this transcript section ({i+1}/{len(chunks)}) of '{video_title}'.
                        STRICT RULES:
                        - Output ONLY dense bullet points.
                        - NO introductory text or conversational filler.
                        - Focus on technical definitions, workflows, and factual data.
                        - Use 'concept: explanation' format for brevity.
                        Transcript:
                        {chunk}
                        """
                    response = llm.invoke(chunk_prompt)
                    partial_summaries.append(response.content)
                    progress_bar.progress((i + 1) / len(chunks))

                combined_notes = "\n\n".join(partial_summaries)
                
                if len(combined_notes) < 12000:
                    final_prompt = f"""
                        Convert these notes into a high-density professional study guide for '{video_title}'.
                        ORGANIZATION RULES:
                        - Use H1 for Title, H2 for Major Themes, and H3 for Sub-topics.
                        - BOLD all key technical terms and formulas.
                        - Remove any redundant or repeating information between sections.
                        - Add a 'Executive Summary' section at the top (max 3 sentences).
                        - End with a 'Key Takeaways' checklist.
                        Raw Notes:
                        {combined_notes}
                        """
                    final_response = llm.invoke(final_prompt)
                    notes_text = final_response.content
                else:
                    notes_text = combined_notes

                st.session_state.notes = notes_text
                st.session_state.title = video_title
                st.success("Notes generated successfully!")

            except Exception as e:
                st.error(f"Error occurred: {e}")

if "notes" in st.session_state:
    st.divider()
    st.markdown(f"### Notes for: {st.session_state.title}")
    st.markdown(st.session_state.notes)
    
    with download_placeholder:
        pdf_data = generate_pdf(st.session_state.notes)
        st.download_button(
            label="📥 Download PDF",
            data=pdf_data,
            file_name=f"{st.session_state.title[:20].replace(' ', '_')}_notes.pdf",
            mime="application/pdf",
            use_container_width=True
        )
            