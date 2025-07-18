import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import tempfile

st.set_page_config(page_title="Smart Summarizer", page_icon="📄")
st.title("📄 PDF & Text Summarizer using LangChain")

# Ask user for Groq API key
st.sidebar.title("🔐 API Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# Initialize LLM with Groq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

input_type = st.radio("Choose Input Type:", ["Write Text", "Upload PDF"])

user_input = ""

if input_type == "Write Text":
    user_input = st.text_area("Enter text to summarize:")
elif input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            reader = PdfReader(tmp_file.name)
            user_input = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

if user_input:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([user_input])

    if len(user_input.split()) < 800:
        # Use Stuff chain for short text
        chain = load_summarize_chain(llm, chain_type="stuff")
    else:
        # Use MapReduce chain for long text
        chain = load_summarize_chain(llm, chain_type="map_reduce")

    with st.spinner("Summarizing..."):
        summary = chain.run(docs)

    st.subheader("📝 Summary")
    st.write(summary)
