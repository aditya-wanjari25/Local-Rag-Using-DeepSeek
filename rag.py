import streamlit as st

st.set_page_config(
    page_title="DeepSeek RAG",
    page_icon="🤖"
)

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os

st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;  /* VS Code dark theme background */
        color: #d4d4d4;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #2d2d2d !important;
        color: #d4d4d4 !important;
        border: 1px solid #454545 !important;
        border-radius: 4px !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }
    
    .stChatInput input:focus {
        border-color: #007acc !important;  /* VS Code blue */
        box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.2) !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #252526 !important;
        border: 1px solid #454545 !important;
        border-radius: 4px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2d2d2d !important;
        border: 1px solid #454545 !important;
        border-radius: 4px !important;
        padding: 15px !important;
        margin: 8px 0 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #007acc !important;  /* VS Code blue */
        color: white !important;
        border-radius: 3px !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #d4d4d4 !important;
    }
    
    .stFileUploader {
        background-color: #2d2d2d;
        border: 1px dashed #454545;
        border-radius: 4px;
        padding: 15px;
    }
    
    .stFileUploader:hover {
        border-color: #007acc;
        background-color: #333333;
    }
    
    h1 {
        color: #569cd6 !important;  /* VS Code blue for keywords */
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-weight: 600 !important;
    }
    
    h2, h3 {
        color: #4ec9b0 !important;  /* VS Code teal for types */
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-weight: 500 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1e3228 !important;  /* Dark green background */
        border: 1px solid #4ec9b0 !important;
        border-radius: 4px !important;
        color: #4ec9b0 !important;
        padding: 12px !important;
    }
    
    /* Spinner styling */
    .stSpinner {
        border-color: #007acc !important;
    }
    
    /* Syntax highlighting colors */
    code {
        color: #ce9178 !important;  /* VS Code orange for strings */
    }
    
    pre {
        background-color: #1e1e1e !important;
        border: 1px solid #454545 !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file):
    # Create the directory if it doesn't exist
    save_dir = "document_store/pdfs"
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("Local RAG using DeepSeek R1-1.5B")
st.markdown("### Smart PDF Agent!")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)