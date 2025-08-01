import os
import fitz  # PyMuPDF
import cohere
import chromadb
from fastapi import FastAPI, Query
from pydantic import BaseModel
from chromadb.config import Settings
from chromadb.utils.embedding_functions import CohereEmbeddingFunction

# === CONFIG ===
COHERE_API_KEY = os.getenv("COHERE_API_KEY") or "your-cohere-key"
POLICY_FOLDER = "policies"
COLLECTION_NAME = "insurance_policies"

# === INIT ===
co = cohere.Client(COHERE_API_KEY)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
embedder = CohereEmbeddingFunction(api_key=COHERE_API_KEY)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)
app = FastAPI()

# === LOAD PDFs ===
def load_pdfs(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(folder_path, file))
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    texts.append({
                        "id": f"{file}-{i}",
                        "text": text,
                        "metadata": {"source": file, "page": i}
                    })
    return texts

# === INGEST ===
def ingest():
    all_chunks = load_pdfs(POLICY_FOLDER)
    ids, docs, meta = [], [], []
    for chunk in all_chunks:
        ids.append(chunk["id"])
        docs.append(chunk["text"])
        meta.append(chunk["metadata"])
    collection.add(documents=docs, metadatas=meta, ids=ids)

# === RUN ON STARTUP ===
ingest()

# === QUERY ENDPOINT ===
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_qna(req: QueryRequest):
    query_embed = embedder.embed_query(req.question)
    results = collection.query(query_embeddings=[query_embed], n_results=3)

    context = "\n".join(results["documents"][0])
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {req.question}\nAnswer:"

    response = co.generate(prompt=prompt, max_tokens=200)
    return {"answer": response.generations[0].text.strip()}

@app.get("/")
def health():
    return {"status": "ok"}
