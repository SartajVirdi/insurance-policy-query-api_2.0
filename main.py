from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# === CONFIGURATION ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in Render environment variables
assert GEMINI_API_KEY, "Missing Gemini API key"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === FASTAPI APP ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === HELPERS ===
def extract_text_from_pdf(file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    reader = PdfReader(tmp_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    os.unlink(tmp_path)
    return text

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def find_relevant_chunks(chunks: List[str], query: str, top_k: int = 3) -> List[str]:
    query_embedding = embedder.encode(query)
    chunk_embeddings = embedder.encode(chunks)
    scores = [(i, float(np.dot(query_embedding, emb))) for i, emb in enumerate(chunk_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, _ in scores[:top_k]]

# === ROUTES ===
@app.post("/ask")
async def ask_question(file: UploadFile = File(...), query: str = ""):
    raw_text = extract_text_from_pdf(file)
    chunks = chunk_text(raw_text)
    relevant_chunks = find_relevant_chunks(chunks, query)
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}"

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}
