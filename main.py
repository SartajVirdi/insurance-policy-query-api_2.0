import os
from fastapi import FastAPI, UploadFile, File
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google import Gemini
from pydantic import BaseModel
from typing import List
import shutil

# Init FastAPI
app = FastAPI()

# Set up Gemini LLM
Settings.llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Document folder
DOCS_DIR = "policies"
os.makedirs(DOCS_DIR, exist_ok=True)

# Load existing documents
def load_index():
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    return VectorStoreIndex.from_documents(documents)

index = load_index()
query_engine = index.as_query_engine()

# Request schema
class QueryRequest(BaseModel):
    question: str

# Routes
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    global index, query_engine
    index = load_index()
    query_engine = index.as_query_engine()
    return {"status": "uploaded", "filename": file.filename}

@app.post("/query")
async def query_docs(request: QueryRequest):
    response = query_engine.query(request.question)
    return {"answer": str(response)}
