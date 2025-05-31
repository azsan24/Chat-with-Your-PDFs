import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Functions ===

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("FAISS.index")

def get_conversational_chain():
    Prompt_Template = """
    Answer the question using only the information from the context below.  
    If the answer is not in the context, reply with: "Answer is not available in the context."  
    Be as detailed as possible.

    Context:  
    {context}

    Question:  
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=Prompt_Template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    if not Path("FAISS.index").exists():
        st.error("Please upload and process PDFs before asking questions.")
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("FAISS.index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response['output_text']

# === Streamlit UI ===

st.set_page_config(page_title="PDF Q&A Chat", layout="wide")
st.title("📄 Chat with Your PDFs")

# Sidebar for file upload and processing
st.sidebar.header("📁 Upload & Process PDFs")
pdf_docs = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if st.sidebar.button("🔄 Process PDFs"):
    if pdf_docs:
        with st.spinner("Processing uploaded PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
            st.sidebar.success("PDFs processed and indexed.")
    else:
        st.sidebar.warning("Please upload at least one PDF.")

# Main section for question/answer
st.markdown("## Ask a Question")
question = st.text_input("Enter your question below:")

if question:
    with st.spinner("Getting answer..."):
        try:
            answer = user_input(question)
            if answer:
                st.markdown("### 🧠 Answer:")
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

