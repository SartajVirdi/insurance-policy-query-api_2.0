from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import fitz  # PyMuPDF
import os

app = FastAPI()

# Make sure this env variable is set in your deployment platform (e.g. Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class HackRxRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- PDF text extractor ---
def extract_text_from_pdf_url(url: str) -> str:
    try:
        pdf_bytes = requests.get(url).content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")

# --- Gemini API call ---
def ask_gemini(question: str, context: str) -> str:
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    # Optimized prompt
    prompt = (
        f"Using only the information from the following insurance policy document, "
        f"answer the question briefly and clearly.\n\n"
        f"Document:\n{context[:20000]}\n\n"
        f"Question: {question}"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Gemini API error: {e}"

# --- Main endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(data: HackRxRequest):
    try:
        full_text = extract_text_from_pdf_url(data.documents)
    except Exception as e:
        return {"answers": [f"Failed to extract PDF: {e}"] * len(data.questions)}
    
    answers = [ask_gemini(q, full_text) for q in data.questions]
    return {"answers": answers}
