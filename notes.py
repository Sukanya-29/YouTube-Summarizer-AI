import streamlit as st
import warnings
from pytubefix import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from fpdf import FPDF
import re


warnings.filterwarnings("ignore") 

groq_api = st.session_state.get("groq_api")
url = st.session_state.get("url")


@st.cache_resource(show_spinner=False)
def get_llm(api_key):
    return init_chat_model(
        model="groq:llama-3.1-8b-instant",
        api_key=api_key
    )

class ElegantPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.is_first_page = True

    def header(self):
        if not self.is_first_page:
            self.set_text_color(140, 140, 140)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, "Study Guide & Transcript Analysis", align="R")
            self.set_y(self.get_y() + 10)
            self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def clean_text(self, text):
        text = text.replace("**", "")
        return re.sub(r'^[#*\-\s]+', '', text).strip()

    def render_document(self, title, text):
        self.set_font("Arial", "B", 24)
        self.set_text_color(20, 30, 45)  # Navy
        self.multi_cell(0, 12, self.clean_text(title), align="L")
        self.set_draw_color(70, 130, 180) 
        self.line(15, self.get_y() + 4, 195, self.get_y() + 4)
        self.ln(12)
        self.is_first_page = False 

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(4)
                continue

            is_h1 = line.startswith("# ")
            is_h2 = line.startswith("## ")
            is_h3 = line.startswith("### ")
            is_bullet = line.startswith("*") or line.startswith("-")
            clean_line = self.clean_text(line)
            if not clean_line: continue

            if is_h1:
                self.set_font("Arial", "B", 18)
                self.set_text_color(20, 30, 45)
                self.multi_cell(0, 10, clean_line, align="L")
                continue
            elif is_h2:
                self.set_font("Arial", "B", 14)
                self.set_text_color(40, 60, 90)
                self.multi_cell(0, 8, clean_line, align="L")
                continue
            elif is_h3:
                self.set_font("Arial", "B", 12)
                self.set_text_color(70, 130, 180)
                self.multi_cell(0, 7, clean_line, align="L")
                continue

            if is_bullet:
                clean_line = f"• {clean_line}"

            self.set_font("Arial", size=10)
            self.set_text_color(20, 20, 20)
            safe_text = clean_line.replace('\u2013', '-').replace('\u2014', '-').replace('\u201c', '"').replace('\u201d', '"')
            safe_text = clean_line.encode('latin-1', 'ignore').decode('latin-1')
            self.multi_cell(0, 6, safe_text, align="J") 
            self.ln(2)

def generate_pdf(text):
    pdf = ElegantPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 20, 15) 
    pdf.add_page()
    video_title = st.session_state.get("title", "Generated Study Guide")
    pdf.render_document(video_title, text)
    
    return bytes(pdf.output())

def clean_raw_llm_text(text):
    text = re.sub(r'\bH[1-3]\.\d+\.?\s*', '', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text

with st.sidebar:
    download_placeholder = st.empty()

st.title("🤖 Youtube RAG system") 

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.subheader("📝 Notes Generator")
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
                    Convert these notes into a high-density, professional study guide for '{video_title}'.
                    CRITICAL FORMATTING & CONTENT RULES:
                    1. NO RAW MARKS: Do not output text containing layout markers like 'H2.1', 'H3.1', or bullet dashes next to headers (e.g., use 'Executive Summary', NOT '- Executive Summary').
                    2. NO MATH SYMBOLS ($): Convert all math formulas into clean plain text. Never use '$'. For example, rewrite '$Y=f(X)$' to 'Y = f(X)' and '$X=\{{x1,x2\}}*' to 'X = {{x1, x2, ...}}'.
                    3. NO REDUNDANCY: Do not repeat definitions. If a concept is explained thoroughly in one section, summarize it tightly or omit it in subsequent lists to ensure high density. Combine repetitive sections (like 'Types of Machine Learning' and 'Machine Learning Types') into a single, cohesive breakdown.
                    4. TYPOGRAPHY: Use standard Markdown formatting rules (# for main title, ## for themes, ### for sub-topics).

                    Raw Notes to Refine:
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
    polished_notes = clean_raw_llm_text(st.session_state.notes)
    st.markdown(polished_notes)
    
    with download_placeholder:
        pdf_data = generate_pdf(polished_notes)
        st.toast("Notes are ready to download! 📄")
        st.download_button(
            label="📥 Download PDF",
            data=pdf_data,
            file_name=f"{st.session_state.title[:20].replace(' ', '_')}_notes.pdf",
            mime="application/pdf",
            use_container_width=True
        )
            
