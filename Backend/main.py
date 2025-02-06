import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from typing import List, Optional
import shutil
from langchain.document_loaders import (
    PyPDFLoader, CSVLoader, JSONLoader, TextLoader, 
    UnstructuredExcelLoader, UnstructuredPowerPointLoader, 
    UnstructuredWordDocumentLoader, UnstructuredEPubLoader
)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import BaseRetriever
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"
)


# Ensure OpenAI API key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Please add it to your .env file.")

# Define upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

ALLOWED_EXTENSIONS = {"pdf", "ppt", "pptx", "txt", "csv", "json", 
                      "xml", "doc", "docx", "odt", "xls", "xlsx", "epub"}

# File extension to LangChain loader mapping
FILE_LOADERS = {
    "pdf": PyPDFLoader,
    "csv": CSVLoader,
    "json": JSONLoader,
    "txt": TextLoader,
    "xls": UnstructuredExcelLoader,
    "xlsx": UnstructuredExcelLoader,
    "ppt": UnstructuredPowerPointLoader,
    "pptx": UnstructuredPowerPointLoader,
    "doc": UnstructuredWordDocumentLoader,
    "docx": UnstructuredWordDocumentLoader,
    "odt": UnstructuredWordDocumentLoader,
    "epub": UnstructuredEPubLoader,
}

# Initialize ChromaDB
CHROMA_DB_PATH = "chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

@app.post("/upload/")
async def upload_and_process_files(
    files: List[UploadFile] = File(None), 
    url: Optional[str] = Form(None)
):
    docs = []
    
    # Debugging print
    print("Received files:", files)
    
    # Process Uploaded Files
    if files:
        for file in files:
            ext = file.filename.split(".")[-1]
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"File type {ext} is not allowed.")

            # Load File using appropriate Loader
            if ext in FILE_LOADERS:
                loader = FILE_LOADERS[ext](file_path)
                docs.extend(loader.load())

    # Process URL if provided
    if url:
        url_loader = WebBaseLoader(url)
        docs.extend(url_loader.load())

    if not docs:
        raise HTTPException(status_code=400, detail="No valid documents found.")

    # Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(docs)

    # Generate Embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma.from_documents(doc_chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    vector_store.persist()

    return {"message": "Files processed and stored successfully in ChromaDB", "num_chunks": len(doc_chunks)}


@app.get("/query/")
async def query_documents(question: str = Query(..., description="User's question about the document")):
    # Load ChromaDB
    embeddings =  OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(question, k=5)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Generate response using OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = prompt.format(question=question, context=docs_content)
    response = llm.invoke(messages)

    return {"answer": response}
