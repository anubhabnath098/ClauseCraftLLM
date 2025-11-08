from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import tempfile
import os
from pdfminer.high_level import extract_text
import numpy as np
import json
import faiss
import pickle
import uuid
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv  # ✅ NEW: Load environment variables

# ---- LOAD ENVIRONMENT VARIABLES ----
load_dotenv()  # Reads variables from .env into os.environ

# ---- CONFIG ----
MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
INDEX_PATH = "clauses.index"
META_PATH = "clauses_meta.pkl"

# ✅ Load Gemini API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing GEMINI_API_KEY in .env file")

LABELS = [
    "Payment Terms",
    "Liability",
    "Confidentiality",
    "Intellectual Property",
    "Termination",
    "Governing Law",
    "Indemnity",
    "Data Protection",
    "Warranty",
    "Force Majeure",
    "Assignment",
    "Service Level Agreement",
    "Non-Compete",
    "Non-Solicitation",
    "Dispute Resolution",
    "Limitation of Liability",
    "Arbitration",
    "Compliance with Laws",
    "Change of Control",
    "Insurance",
    "Notices",
    "Representations and Warranties",
    "Amendments",
    "Entire Agreement",
    "Severability",
    "Other"
]



# ---- INITIALIZE ----
print("🔹 Loading embedding model...")
embed_model = SentenceTransformer(MODEL_EMBED)

# Initialize Gemini client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Connected to Google Gemini API successfully.")
except Exception as e:
    print(f"⚠️ Gemini connection failed: {e}")
    model = None

app = FastAPI(title="PDF Link Clause Classifier + Vector DB Builder (Gemini Edition)")

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ Models loaded successfully")

# ---- INPUT SCHEMA ----
class PDFInput(BaseModel):
    url: str

# ---- HELPERS ----
def download_pdf(url: str):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    return tmp.name


def extract_text_from_pdf(path: str):
    try:
        text = extract_text(path)
        return text if text else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


def extract_and_classify_clauses_with_gemini(text: str):
    """
    Use Gemini LLM to extract clauses and classify them.
    Returns a list of dicts with 'text' and 'type'.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Gemini API not configured.")

    prompt = f"""
You are a legal contract analyst. Extract all distinct clauses from the following contract text and classify each one.

For each clause you identify:
1. Extract the complete clause text.
2. Classify it into one of these categories: {', '.join(LABELS)}.

Return your response as a **valid JSON array** where each item has:
- "text": the full clause text
- "type": the classification label

Contract text:
\"\"\"{text[:15000]}\"\"\"

Return only valid JSON — no markdown, no commentary, no explanation.
"""

    try:
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Remove code block formatting if any
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        clauses = json.loads(content)

        if not isinstance(clauses, list):
            raise ValueError("Response is not a JSON list")

        validated_clauses = []
        for clause in clauses:
            if isinstance(clause, dict) and "text" in clause and "type" in clause:
                clause_type = clause["type"]
                if clause_type not in LABELS:
                    clause_type = "Other"
                validated_clauses.append({
                    "text": clause["text"],
                    "type": clause_type
                })

        return validated_clauses

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        print(f"Raw content: {content[:500]}")
        raise HTTPException(status_code=500, detail="Gemini returned invalid JSON.")
    except Exception as e:
        print(f"⚠️ Gemini LLM failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")


def get_embedding(text: str):
    """Generate normalized embedding for text"""
    emb = embed_model.encode(text, convert_to_numpy=True)
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")


def save_faiss(vectors, metas):
    """Save FAISS index and metadata"""
    if len(vectors) == 0:
        raise ValueError("No vectors to save")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    print(f"✅ Saved {len(vectors)} vectors to {INDEX_PATH}")


# ---- MAIN ENDPOINT ----
@app.post("/process_pdf_link/")
def process_pdf_link(data: PDFInput = Body(...)):
    """
    Download PDF, extract and classify clauses using Gemini,
    create embeddings, and store in FAISS.
    """
    pdf_path = download_pdf(data.url)

    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")

        print(f"📄 Extracted {len(text)} characters from PDF")

        clauses = extract_and_classify_clauses_with_gemini(text)

        if not clauses:
            raise HTTPException(status_code=500, detail="Failed to extract clauses")

        print(f"🧩 Extracted {len(clauses)} clauses")

        vectors, metas = [], []
        for clause in clauses:
            cid = str(uuid.uuid4())
            clause_text = clause["text"]
            clause_type = clause["type"]

            emb = get_embedding(clause_text)
            vectors.append(emb)

            metas.append({
                "id": cid,
                "text": clause_text,
                "predicted_type": clause_type,
                "source": data.url
            })

            print(f"✓ {clause_type}: {clause_text[:80]}...")

        vectors = np.vstack(vectors)
        save_faiss(vectors, metas)

        return {
            "status": "success",
            "indexed": len(clauses),
            "index_path": INDEX_PATH,
            "meta_path": META_PATH,
            "clauses": [
                {
                    "id": meta["id"],
                    "type": meta["predicted_type"],
                    "text": meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"]
                }
                for meta in metas
            ]
        }

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ---- SEARCH ENDPOINT ----
@app.post("/search_clauses/")
def search_clauses(query: str = Body(..., embed=True), top_k: int = Body(5, embed=True)):
    """
    Search for similar clauses in the FAISS database
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise HTTPException(status_code=404, detail="No index found. Process a PDF first.")

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)

    query_emb = get_embedding(query)
    query_emb = np.expand_dims(query_emb, 0)

    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metas):
            results.append({
                "rank": i + 1,
                "distance": float(distances[0][i]),
                "clause": metas[idx]
            })

    return {"query": query, "results": results}
