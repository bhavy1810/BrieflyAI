# =========================================================
# BrieflyAI – RAG-based PDF, DOCX & Text Summarizer
# LangChain + Groq + FAISS
# =========================================================

import streamlit as st
import os
import io
from io import BytesIO
from dotenv import load_dotenv

# PDF
try:
    import PyPDF2
except ImportError:
    import pypdf as PyPDF2

# DOCX
from docx import Document

# PDF Export
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================================================
# CONFIG
# =========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="BrieflyAI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 BrieflyAI")
st.markdown("Upload **PDF / DOCX** or paste text to get **RAG-powered AI summaries** 🚀")

# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "quick_summary": None,
    "detailed_summary": None,
    "vectorstore": None,
    "summary_type": None,
    "is_generating": False,
    "source_text": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================
# LLM
# =========================================================
# Ensure API Key is available
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables.")
    st.stop()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)
output_parser = StrOutputParser()

# =========================================================
# HELPERS
# =========================================================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)

def rag_summary(vectorstore, summary_type):
    if summary_type == "quick":
        k = 4
        max_words = 200
        template = """
You are an expert summarization AI.
Using ONLY the context below, generate a clear and concise summary of about {max_words} words.

Context:
{context}

Quick Summary:
"""
    else:
        k = 10
        max_words = 800
        template = """
You are an expert summarization AI.
Using ONLY the context below, generate a detailed summary of about {max_words} words.
Use headings, bullet points, and clear explanations.

Context:
{context}

Detailed Summary:
"""

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke("Summarize the document")
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | output_parser

    # Truncate context to avoid token limits if necessary
    return chain.invoke({
        "context": context[:25000],
        "max_words": max_words
    })

# =========================================================
# EXPORTS
# =========================================================
def export_pdf(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Handle line breaks for PDF
    for line in text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def export_docx(text):
    buffer = BytesIO()
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# =========================================================
# UI
# =========================================================
tab1, tab2 = st.tabs(["📤 Upload File", "✍️ Paste Text"])

with tab1:
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    col1, col2 = st.columns(2)

    if uploaded:
        if uploaded.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded)
        else:
            text = extract_text_from_docx(uploaded)

        with col1:
            if st.button("⚡ Quick Summary", key="btn_quick_file", use_container_width=True):
                if not st.session_state.is_generating:
                    st.session_state.is_generating = True
                    st.session_state.source_text = text
                    with st.spinner("Indexing & Summarizing..."):
                        st.session_state.vectorstore = build_vectorstore(text)
                        st.session_state.quick_summary = rag_summary(
                            st.session_state.vectorstore, "quick"
                        )
                    st.session_state.summary_type = "quick"
                    st.session_state.is_generating = False
                    st.rerun()

        with col2:
            if st.button("📋 Detailed Summary", key="btn_detailed_file", use_container_width=True):
                if not st.session_state.is_generating:
                    st.session_state.is_generating = True
                    st.session_state.source_text = text
                    with st.spinner("Indexing & Summarizing..."):
                        st.session_state.vectorstore = build_vectorstore(text)
                        st.session_state.detailed_summary = rag_summary(
                            st.session_state.vectorstore, "detailed"
                        )
                    st.session_state.summary_type = "detailed"
                    st.session_state.is_generating = False
                    st.rerun()

with tab2:
    pasted = st.text_area("Paste text", height=300)
    col1, col2 = st.columns(2)

    with col1:
        if pasted and st.button("⚡ Quick Summary", key="btn_quick_paste", use_container_width=True):
            if not st.session_state.is_generating:
                st.session_state.is_generating = True
                st.session_state.source_text = pasted
                with st.spinner("Indexing & Summarizing..."):
                    st.session_state.vectorstore = build_vectorstore(pasted)
                    st.session_state.quick_summary = rag_summary(
                        st.session_state.vectorstore, "quick"
                    )
                st.session_state.summary_type = "quick"
                st.session_state.is_generating = False
                st.rerun()

    with col2:
        if pasted and st.button("📋 Detailed Summary", key="btn_detailed_paste", use_container_width=True):
            if not st.session_state.is_generating:
                st.session_state.is_generating = True
                st.session_state.source_text = pasted
                with st.spinner("Indexing & Summarizing..."):
                    st.session_state.vectorstore = build_vectorstore(pasted)
                    st.session_state.detailed_summary = rag_summary(
                        st.session_state.vectorstore, "detailed"
                    )
                st.session_state.summary_type = "detailed"
                st.session_state.is_generating = False
                st.rerun()

# =========================================================
# RESULT
# =========================================================
summary = (
    st.session_state.quick_summary
    if st.session_state.summary_type == "quick"
    else st.session_state.detailed_summary
)

if summary:
    st.divider()
    st.subheader("🧠 Summary")
    
    # 1. Render the Markdown Summary (Visual)
    st.markdown(summary)

    # 2. FIXED: Robust Copy Functionality
    # Using st.code provides a native, 100% working copy button in the top right.
    st.write("---")
    with st.expander("📝 View Raw Text (Copy to Clipboard)"):
        st.code(summary, language="markdown")

    # 3. Exports
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Export TXT",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📕 Export PDF",
            data=export_pdf(summary),
            file_name="summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col3:
        st.download_button(
            label="📝 Export DOCX",
            data=export_docx(summary),
            file_name="summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown("Built with ❤️ using **RAG • LangChain • Groq**")