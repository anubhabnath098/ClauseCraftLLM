# analyze_contract_app.py
"""
Analyze a new contract against an existing Playbook FAISS index.

Features:
- Accepts a contract by URL or raw text.
- Extracts clauses + labels via Google Gemini.
- Embeds clauses with a sentence-transformer and queries existing FAISS playbook index.
- Optionally uses Gemini to check semantic contradictions between new clause and top-k playbook matches.

Install:
pip install fastapi uvicorn sentence-transformers google-generativeai faiss-cpu numpy requests pdfminer.six python-dotenv

Run:
uvicorn analyze_contract_app:app --reload
"""

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import requests
import tempfile
import json
import re
import uuid
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()  # loads variables from .env

MODEL_EMBED = os.getenv("MODEL_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
PLAYBOOK_INDEX_PATH = os.getenv("PLAYBOOK_INDEX_PATH", "clauses.index")
PLAYBOOK_META_PATH = os.getenv("PLAYBOOK_META_PATH", "clauses_meta.pkl")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

ORIGINS = ["http://localhost:3000"]

# ---------------- INIT ----------------
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not set. Gemini LLM calls will be disabled.")
    gemini_model = None
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini configured for model {GEMINI_MODEL}")
    except Exception as e:
        print("⚠️ Failed to initialize Gemini client:", e)
        gemini_model = None

print("🔹 Loading embedding model:", MODEL_EMBED)
embed_model = SentenceTransformer(MODEL_EMBED)
print("✅ Embedding model loaded")

# ---------------- FASTAPI ----------------
app = FastAPI(title="Contract -> Playbook Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SCHEMAS ----------------
class AnalyzeInput(BaseModel):
    url: Optional[str] = None
    raw_text: Optional[str] = None
    top_k: Optional[int] = 5
    use_llm_contradiction_check: Optional[bool] = True

# ---------------- HELPERS ----------------
def download_pdf(url: str, timeout: int = 30) -> str:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name

def extract_text_from_pdf(path: str) -> str:
    try:
        text = extract_text(path) or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

def clean_text_singleline(t: str) -> str:
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def get_embedding(text: str) -> np.ndarray:
    emb = embed_model.encode(text, convert_to_numpy=True)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb.astype("float32")
    return (emb / norm).astype("float32")

# ---------------- Gemini clause extraction ----------------
def extract_and_classify_clauses_with_gemini(text: str, max_chars: int = 15000) -> List[Dict[str,str]]:
    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini client not configured.")
    prompt = f"""
You are a legal contract analyst. Extract distinct clauses from the contract text below and classify each with a short label.

Return ONLY valid JSON array with:
- "text": full clause
- "type": short label (Payment Terms, Confidentiality, Liability, etc.)

Contract text:
\"\"\"{text[:max_chars]}\"\"\"
"""
    try:
        resp = gemini_model.generate_content(prompt)
        content = resp.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini extraction failed: {e}")

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        parsed = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini JSON. Raw: {content[:1200]}")

    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        t = item.get("text") or item.get("clause") or item.get("clause_text")
        typ = item.get("type") or item.get("label") or "Other"
        if t:
            validated.append({"text": t.strip(), "type": typ.strip() if isinstance(typ, str) else "Other"})
    return validated

# ---------------- Gemini contradiction check ----------------
def check_contradiction_with_gemini(new_clause: str, ref_clause: str) -> Dict[str, Any]:
    if gemini_model is None:
        return {"contradiction": None, "explanation": "Gemini not configured"}
    prompt = f"""
Compare Clause A (new) and Clause B (playbook). Return JSON: {{"contradiction": <true|false>, "explanation": "<reason>"}}

Clause A:
\"\"\"{new_clause}\"\"\"

Clause B:
\"\"\"{ref_clause}\"\"\"
"""
    try:
        resp = gemini_model.generate_content(prompt)
        txt = resp.text.strip()
    except Exception as e:
        return {"contradiction": None, "explanation": f"Gemini call failed: {e}"}

    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
    if txt.endswith("```"):
        txt = txt[:-3].strip()

    try:
        j = json.loads(txt)
        return {
            "contradiction": bool(j.get("contradiction")) if "contradiction" in j else None,
            "explanation": str(j.get("explanation", "")).strip()
        }
    except Exception:
        return {"contradiction": None, "explanation": f"Could not parse Gemini response. Raw: {txt[:400]}"}

# ---------------- Endpoint: analyze contract ----------------
from fastapi import FastAPI, Body, HTTPException, File, UploadFile

# ... (imports remain the same)

# ... (FastAPI app setup remains the same)

# ---------------- SCHEMAS ----------------
class AnalyzeInput(BaseModel):
    url: Optional[str] = None
    raw_text: Optional[str] = None
    top_k: Optional[int] = 5
    use_llm_contradiction_check: Optional[bool] = True

# ... (helper functions remain the same)

# ---------------- Endpoint: analyze contract ----------------
@app.post("/analyze_contract/")
async def analyze_contract(file: UploadFile = File(None), payload: AnalyzeInput = Body(None)):
    if file:
        pdf_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                pdf_path = tmp.name
            contract_text = extract_text_from_pdf(pdf_path)
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
    elif payload and (payload.raw_text or payload.url):
        contract_text = payload.raw_text
        if not contract_text:
            pdf_path = download_pdf(payload.url)
            try:
                contract_text = extract_text_from_pdf(pdf_path)
            finally:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
    else:
        raise HTTPException(status_code=400, detail="Provide either a file, raw_text, or url")

    if not os.path.exists(PLAYBOOK_INDEX_PATH) or not os.path.exists(PLAYBOOK_META_PATH):
        raise HTTPException(status_code=404, detail="Playbook FAISS index or metadata not found.")

    # ... (rest of the function remains the same)
    # Load FAISS index + metadata
    try:
        index = faiss.read_index(PLAYBOOK_INDEX_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read FAISS index: {e}")
    try:
        with open(PLAYBOOK_META_PATH, "rb") as f:
            metas = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load playbook metadata: {e}")

    if not contract_text or not contract_text.strip():
        raise HTTPException(status_code=400, detail="No extractable contract text")

    # Extract clauses
    new_clauses = extract_and_classify_clauses_with_gemini(contract_text)
    if not new_clauses:
        return {"status": "success", "checked": [], "message": "No clauses extracted"}

    session_id = f"session_{int(uuid.uuid4().int >> 64)}"
    output_clauses = []
    suggestions = []

    for clause in new_clauses:
        text = clause["text"]
        typ = clause.get("type", "Other")
        emb = get_embedding(text).reshape(1, -1)

        # FAISS search
        D, I = index.search(emb, payload.top_k if payload else 5)
        top_matches = []
        any_contradiction = False

        for rank_idx, ref_idx in enumerate(I[0]):
            if ref_idx >= len(metas):
                continue
            ref = metas[ref_idx]
            check = check_contradiction_with_gemini(text, ref.get("text")) if payload and payload.use_llm_contradiction_check else {"contradiction": None, "explanation": "LLM disabled"}
            if check.get("contradiction"):
                any_contradiction = True

            top_matches.append({
                "rank": int(rank_idx + 1),
                "reference_type": ref.get("predicted_type"),
                "reference_text": ref.get("text"),
                "contradiction_check": check
            })

        output_clauses.append({
            "title": typ,
            "content": text,
            "riskLevel": "high" if any_contradiction else "medium"
        })

        suggestions.append({
            "clause": typ,
            "suggestion": f"Review based on top {payload.top_k if payload else 5} similar clauses from playbook.",
            "priority": "high" if any_contradiction else "medium"
        })

    return {
        "sessionId": session_id,
        "clauses": output_clauses,
        "suggestions": suggestions,
        "timestamp": f"{np.datetime64('now')}"
    }

# ---------------- Health ----------------

@app.get("/health")

def health():

    return {

        "status": "ok",

        "gemini_configured": gemini_model is not None,

        "playbook_index_exists": os.path.exists(PLAYBOOK_INDEX_PATH) and os.path.exists(PLAYBOOK_META_PATH)

    }



# ---------------- Chat ----------------

class ChatInput(BaseModel):

    message: str

    session_id: str



@app.post("/chat/")

def chat(payload: ChatInput = Body(...)):

    # In a real application, you would use the session_id to retrieve context

    # and then call a language model with the message and context.

    return {"response": f"This is a dummy response to your message: '{payload.message}'"}
