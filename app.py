# =========================================================
# BrieflyAI – RAG-based PDF & Text Summarizer
# LangChain + Groq + FAISS
# =========================================================
import streamlit as st
import os
import io
from dotenv import load_dotenv

try:
    import PyPDF2
except ImportError:
    import pypdf as PyPDF2

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
st.markdown("Upload PDFs or paste text to get **RAG-powered AI summaries** 🚀")

# =========================================================
# SESSION STATE
# =========================================================
for key in ["quick_summary", "detailed_summary", "is_generating", "vectorstore", "summary_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================================
# LLM SETUP
# =========================================================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

output_parser = StrOutputParser()

# =========================================================
# HELPERS
# =========================================================
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except:
        return None

def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)

def rag_summary(vectorstore, summary_type="quick"):
    """Generate either quick or detailed summary"""
    
    if summary_type == "quick":
        k_chunks = 4
        max_words = 200
        prompt_template = """You are an expert summarization AI. 
Using ONLY the context below, generate a clear and concise summary of approximately {max_words} words.
Focus on the main points and key takeaways.

Context:
{context}

Quick Summary:"""
    else:  # detailed
        k_chunks = 10
        max_words = 800
        prompt_template = """You are an expert summarization AI.
Using ONLY the context below, generate a comprehensive and detailed summary of approximately {max_words} words.
Include:
- Main themes and key points
- Important details and supporting information
- Relevant examples or data
- Logical structure with clear sections

Context:
{context}

Detailed Summary:"""
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_chunks})
    context_docs = retriever.invoke("Summarize the entire document")
    context = "\n\n".join(doc.page_content for doc in context_docs)
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | output_parser
    
    return chain.invoke({
        "context": context[:25_000],
        "max_words": max_words
    })

# =========================================================
# UI
# =========================================================
tab1, tab2 = st.tabs(["📤 Upload PDF", "✍️ Paste Text"])

# ---------------- PDF TAB ----------------
with tab1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file and st.button("⚡ Quick Summary", type="primary", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Missing GROQ_API_KEY")
            else:
                st.session_state.is_generating = True
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    with st.spinner("🔎 Building knowledge base..."):
                        st.session_state.vectorstore = build_vectorstore(text)
                    with st.spinner("🤖 Generating quick summary..."):
                        st.session_state.quick_summary = rag_summary(
                            st.session_state.vectorstore,
                            summary_type="quick"
                        )
                        st.session_state.summary_type = "quick"
                    st.session_state.is_generating = False
                    st.rerun()
    
    with col2:
        if uploaded_file and st.button("📋 Detailed Summary", type="secondary", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Missing GROQ_API_KEY")
            else:
                st.session_state.is_generating = True
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    with st.spinner("🔎 Building knowledge base..."):
                        if not st.session_state.vectorstore:
                            st.session_state.vectorstore = build_vectorstore(text)
                    with st.spinner("🤖 Generating detailed summary..."):
                        st.session_state.detailed_summary = rag_summary(
                            st.session_state.vectorstore,
                            summary_type="detailed"
                        )
                        st.session_state.summary_type = "detailed"
                    st.session_state.is_generating = False
                    st.rerun()

# ---------------- TEXT TAB ----------------
with tab2:
    pasted_text = st.text_area("Paste your text here", height=300)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if pasted_text and st.button("⚡ Quick Summary ", type="primary", use_container_width=True, key="quick_text"):
            st.session_state.is_generating = True
            with st.spinner("🔎 Building knowledge base..."):
                st.session_state.vectorstore = build_vectorstore(pasted_text)
            with st.spinner("🤖 Generating quick summary..."):
                st.session_state.quick_summary = rag_summary(
                    st.session_state.vectorstore,
                    summary_type="quick"
                )
                st.session_state.summary_type = "quick"
            st.session_state.is_generating = False
            st.rerun()
    
    with col2:
        if pasted_text and st.button("📋 Detailed Summary ", type="secondary", use_container_width=True, key="detailed_text"):
            st.session_state.is_generating = True
            with st.spinner("🔎 Building knowledge base..."):
                if not st.session_state.vectorstore:
                    st.session_state.vectorstore = build_vectorstore(pasted_text)
            with st.spinner("🤖 Generating detailed summary..."):
                st.session_state.detailed_summary = rag_summary(
                    st.session_state.vectorstore,
                    summary_type="detailed"
                )
                st.session_state.summary_type = "detailed"
            st.session_state.is_generating = False
            st.rerun()

# =========================================================
# RESULT
# =========================================================
if st.session_state.quick_summary or st.session_state.detailed_summary:
    st.divider()
    
    # Display both summaries if available
    if st.session_state.quick_summary and st.session_state.detailed_summary:
        summary_tab1, summary_tab2 = st.tabs(["⚡ Quick Summary", "📋 Detailed Summary"])
        
        with summary_tab1:
            st.markdown(st.session_state.quick_summary)
            st.download_button(
                "📥 Download Quick Summary",
                st.session_state.quick_summary,
                "quick_summary.txt",
                "text/plain",
                use_container_width=True,
                key="download_quick"
            )
        
        with summary_tab2:
            st.markdown(st.session_state.detailed_summary)
            st.download_button(
                "📥 Download Detailed Summary",
                st.session_state.detailed_summary,
                "detailed_summary.txt",
                "text/plain",
                use_container_width=True,
                key="download_detailed"
            )
    
    # Display single summary
    elif st.session_state.summary_type == "quick" and st.session_state.quick_summary:
        st.subheader("⚡ Quick Summary")
        st.markdown(st.session_state.quick_summary)
        st.download_button(
            "📥 Download Summary",
            st.session_state.quick_summary,
            "quick_summary.txt",
            "text/plain",
            use_container_width=True
        )
    
    elif st.session_state.summary_type == "detailed" and st.session_state.detailed_summary:
        st.subheader("📋 Detailed Summary")
        st.markdown(st.session_state.detailed_summary)
        st.download_button(
            "📥 Download Summary",
            st.session_state.detailed_summary,
            "detailed_summary.txt",
            "text/plain",
            use_container_width=True
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(
    "Built with ❤️ using RAG • LangChain • Groq",
    unsafe_allow_html=True
)