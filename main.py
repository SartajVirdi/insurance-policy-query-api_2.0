from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import re
import glob
from sentence_transformers import SentenceTransformer
import cohere

app = Flask(__name__)

# ✅ Load embedding model (lightweight)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Optionally, load Cohere (hosted LLM)
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# ✅ Load and combine all PDFs from the 'policies/' folder
def load_all_pdfs(folder="policies"):
    combined_text = ""
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        doc = fitz.open(filepath)
        for page in doc:
            combined_text += page.get_text() + "\n"
    return combined_text

# ✅ Smarter chunking with larger context and overlap
def split_text(text, max_tokens=400, overlap=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    chunk = []
    tokens = 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if tokens + sentence_tokens <= max_tokens:
            chunk.append(sentence)
            tokens += sentence_tokens
        else:
            chunks.append(" ".join(chunk))
            chunk = chunk[-(overlap//5):] + [sentence]  # maintain overlap
            tokens = sum(len(s.split()) for s in chunk)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Query field missing"}), 400

    query = data["query"]

    try:
        text = load_all_pdfs()
    except Exception as e:
        return jsonify({"error": f"Failed to load PDFs: {e}"}), 500

    chunks = split_text(text)
    try:
        embeddings = embedder.encode(chunks)
    except Exception as e:
        return jsonify({"error": f"Embedding error: {e}"}), 500

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    try:
        query_embed = embedder.encode([query])[0]
        D, I = index.search(np.array([query_embed]), k=5)
        retrieved = [chunks[i] for i in I[0]]
    except Exception as e:
        return jsonify({"error": f"Search error: {e}"}), 500

    context = "\n".join(retrieved)

    try:
        response = co.chat(
            model='command-r-plus',
            message=query,
            documents=[{"text": context}]
        )
        return jsonify({"response": response.text.strip()})
    except Exception as e:
        return jsonify({"error": f"LLM response error: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
