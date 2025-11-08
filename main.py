# analyze_contract_app.py
"""
Analyze a new contract against an existing Playbook FAISS index.

Features:
- Accepts a contract by URL or raw text.
- Extracts clauses + labels via Google Gemini.
- Embeds clauses with a sentence-transformer and queries existing FAISS playbook index.
- Optionally uses Gemini to check semantic contradictions between new clause and top-k playbook matches.
- Dynamically generates AI-driven redline suggestions & rationale.

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
app = FastAPI(title="Contract → Playbook Analyzer")
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

# ---------------- Gemini dynamic suggestion generation ----------------
def generate_suggestion_for_clause(new_clause: str, top_matches: List[Dict[str, Any]], max_chars: int = 12000) -> Dict[str, Any]:
    """
    Compare a new clause to top playbook matches and generate JSON with:
    - recommended_change
    - suggested_redline
    - rationale
    - severity
    - differences
    """
    if gemini_model is None:
        return {"note": "Gemini not configured", "raw": None}

    matches_text = ""
    for i, m in enumerate(top_matches, start=1):
        excerpt = m.get("reference_text", "") or ""
        if len(excerpt) > 1000:
            excerpt = excerpt[:1000] + " ... [truncated]"
        matches_text += f"{i}. (Type: {m.get('reference_type','')}) {excerpt}\n\n"

    prompt = f"""
You are an expert contract reviewer. Compare Clause A (new) with the following top playbook clauses and produce a clear improvement suggestion.

Return ONLY strict JSON with:
- "recommended_change": short summary of what should be changed
- "suggested_redline": proposed concise redline or rewrite
- "rationale": short explanation referencing key differences
- "severity": "low", "medium", or "high" risk
- "differences": array of short bullet points listing main differences

Clause A:
\"\"\"{new_clause[:max_chars]}\"\"\"

Top playbook clauses:
\"\"\"{matches_text}\"\"\"
"""

    try:
        resp = gemini_model.generate_content(prompt)
        txt = (resp.text or "").strip()
    except Exception as e:
        return {"note": f"Gemini call failed: {e}", "raw": None}

    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
    if txt.endswith("```"):
        txt = txt[:-3].strip()

    try:
        parsed = json.loads(txt)
        return {
            "recommended_change": parsed.get("recommended_change"),
            "suggested_redline": parsed.get("suggested_redline"),
            "rationale": parsed.get("rationale"),
            "severity": parsed.get("severity"),
            "differences": parsed.get("differences") if isinstance(parsed.get("differences"), list) else []
        }
    except Exception:
        return {"note": "Could not parse LLM JSON", "raw": txt}

# ---------------- Endpoint: analyze contract ----------------
@app.post("/analyze_contract/")
def analyze_contract(payload: AnalyzeInput = Body(...)):
    if not payload.raw_text and not payload.url:
        raise HTTPException(status_code=400, detail="Provide either raw_text or url")

    if not os.path.exists(PLAYBOOK_INDEX_PATH) or not os.path.exists(PLAYBOOK_META_PATH):
        raise HTTPException(status_code=404, detail="Playbook FAISS index or metadata not found.")

    try:
        index = faiss.read_index(PLAYBOOK_INDEX_PATH)
        with open(PLAYBOOK_META_PATH, "rb") as f:
            metas = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index/meta: {e}")

    # get contract text
    pdf_path = None
    contract_text = payload.raw_text
    if not contract_text:
        pdf_path = download_pdf(payload.url)
        try:
            contract_text = extract_text_from_pdf(pdf_path)
        finally:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

    if not contract_text.strip():
        raise HTTPException(status_code=400, detail="No extractable contract text")

    # extract clauses
    new_clauses = extract_and_classify_clauses_with_gemini(contract_text)
    if not new_clauses:
        return {"status": "success", "checked": [], "message": "No clauses extracted"}

    session_id = f"session_{int(uuid.uuid4().int >> 64)}"
    output_clauses = []
    suggestions = []
    top_k = int(payload.top_k or 5)

    # --- Process each clause ---
    for clause in new_clauses:
        text = clause["text"]
        typ = clause.get("type", "Other")

        emb = get_embedding(text).reshape(1, -1)
        D, I = index.search(emb, top_k)

        top_matches = []
        any_contradiction = False

        for rank_idx, ref_idx in enumerate(I[0]):
            if ref_idx < 0 or ref_idx >= len(metas):
                continue
            ref = metas[ref_idx]
            check = (
                check_contradiction_with_gemini(text, ref.get("text"))
                if payload.use_llm_contradiction_check
                else {"contradiction": None, "explanation": "LLM disabled"}
            )
            if check.get("contradiction"):
                any_contradiction = True
            top_matches.append({
                "rank": int(rank_idx + 1),
                "reference_type": ref.get("predicted_type"),
                "reference_text": ref.get("text"),
                "distance": float(D[0][rank_idx]) if D is not None else None,
                "contradiction_check": check
            })

        llm_suggestion = generate_suggestion_for_clause(text, top_matches)

        severity = llm_suggestion.get("severity") if isinstance(llm_suggestion, dict) else None
        risk_level = severity if severity in ["low", "medium", "high"] else ("high" if any_contradiction else "medium")

        output_clauses.append({
            "title": typ,
            "content": text,
            "riskLevel": risk_level,
            "top_matches_count": len(top_matches)
        })

        suggestions.append({
            "clause": typ,
            "priority": risk_level,
            "llm_suggestion": llm_suggestion,
            "top_matches": top_matches
        })

    return {
        "sessionId": session_id,
        "clauses": output_clauses,
        "suggestions": suggestions,
        "timestamp": f"{np.datetime64('now')}"
    }

# ---------------- Chat endpoint ----------------
class ChatInput(BaseModel):
    sessionId: str
    message: str
    top_k: Optional[int] = 5

# simple in-memory session store for conversation continuity
sessions: Dict[str, Dict[str, Any]] = {}

@app.post("/chat/")
def chat_endpoint(payload: ChatInput = Body(...)):
    """
    Chat endpoint:
    - Takes user's question/message
    - Retrieves relevant clauses from FAISS
    - Uses Gemini to answer conversationally
    """

    if not os.path.exists(PLAYBOOK_INDEX_PATH) or not os.path.exists(PLAYBOOK_META_PATH):
        raise HTTPException(status_code=404, detail="Playbook FAISS index or metadata not found.")

    # Load vector index + metadata
    try:
        index = faiss.read_index(PLAYBOOK_INDEX_PATH)
        with open(PLAYBOOK_META_PATH, "rb") as f:
            metas = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load FAISS or metadata: {e}")

    query = payload.message.strip()
    top_k = int(payload.top_k or 5)

    if not query:
        raise HTTPException(status_code=400, detail="Empty message provided")

    # Compute query embedding
    qv = get_embedding(query).reshape(1, -1)
    D, I = index.search(qv, top_k)
    retrieved = []
    for rank_idx, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        meta = metas[idx]
        retrieved.append({
            "rank": int(rank_idx + 1),
            "text": meta.get("text"),
            "predicted_type": meta.get("predicted_type"),
            "distance": float(D[0][rank_idx])
        })

    # build context for LLM
    context_text = ""
    for i, r in enumerate(retrieved, start=1):
        excerpt = r["text"]
        if len(excerpt) > 800:
            excerpt = excerpt[:800] + " ... [truncated]"
        context_text += f"{i}. ({r['predicted_type']}) {excerpt}\n"

    prompt = f"""
You are a legal contract assistant. Use the following relevant clauses from the playbook to answer user questions accurately.

Contract reference excerpts:
{context_text}

User message:
"{query}"

Instructions:
- Respond conversationally and clearly.
- Use the contract excerpts only when relevant.
- If unsure, explain possible interpretations.
- Keep the tone professional and concise.
"""

    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini client not configured.")

    try:
        response = gemini_model.generate_content(prompt)
        reply_text = response.text.strip() if response and response.text else "(No reply)"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini chat call failed: {e}")

    # store minimal chat history
    sid = payload.sessionId
    if sid not in sessions:
        sessions[sid] = {"history": []}
    sessions[sid]["history"].append({"role": "user", "message": query})
    sessions[sid]["history"].append({"role": "assistant", "message": reply_text})

    return {
        "sessionId": sid,
        "response": reply_text,
        "retrieved_clauses": retrieved,
        "conversation_length": len(sessions[sid]["history"]),
    }

# ---------------- Negotiation endpoint ----------------
class NegotiationInput(BaseModel):
    sessionId: Optional[str] = None
    message: str
    style: str  # "aggressive" | "mildly_aggressive" | "friendly"
    context: str
    gender: str  # "male" | "female"
    top_k: Optional[int] = 5


@app.post("/negotiate/")
def negotiate(payload: NegotiationInput = Body(...)):
    """
    Multi-turn AI negotiation endpoint:
    - Takes user message + style + context + gender
    - Fetches relevant clauses via FAISS
    - Uses Gemini to simulate a negotiation tone
    - Maintains in-memory session chat history
    """

    if not os.path.exists(PLAYBOOK_INDEX_PATH) or not os.path.exists(PLAYBOOK_META_PATH):
        raise HTTPException(status_code=404, detail="Playbook FAISS index or metadata not found.")

    # Load FAISS + metadata
    try:
        index = faiss.read_index(PLAYBOOK_INDEX_PATH)
        with open(PLAYBOOK_META_PATH, "rb") as f:
            metas = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load FAISS or metadata: {e}")

    query = payload.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty message provided")

    top_k = int(payload.top_k or 5)
    qv = get_embedding(query).reshape(1, -1)
    D, I = index.search(qv, top_k)
    retrieved = []
    for rank_idx, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(metas):
            continue
        meta = metas[idx]
        retrieved.append({
            "rank": int(rank_idx + 1),
            "text": meta.get("text"),
            "predicted_type": meta.get("predicted_type"),
            "distance": float(D[0][rank_idx])
        })

    context_text = ""
    for i, r in enumerate(retrieved, start=1):
        excerpt = r["text"]
        if len(excerpt) > 800:
            excerpt = excerpt[:800] + " ... [truncated]"
        context_text += f"{i}. ({r['predicted_type']}) {excerpt}\n"

    # Maintain conversation context
    sid = payload.sessionId or f"session_{uuid.uuid4().hex[:10]}"
    if sid not in sessions:
        sessions[sid] = {"history": []}

    full_context = (
        "Negotiation context: " + payload.context + "\n\n" +
        "Conversation so far:\n" +
        "\n".join([f"{m['role']}: {m['message']}" for m in sessions[sid]["history"][-5:]])  # last 5 turns
    )

    # Build the negotiation prompt for Gemini
    prompt = f"""
You are an expert **contract negotiator AI** simulating a real conversation.
Respond briefly (1–2 sentences) in a {payload.style.replace("_", " ")} and professional tone, 
using persuasive negotiation language appropriate to the given context.

Playbook reference clauses:
{context_text}

Negotiation context:
{payload.context}

Conversation so far:
{full_context}

User ({payload.gender} voice): "{query}"

Your task:
1. Give a short and realistic reply continuing the negotiation naturally.
2. After your reply, ALWAYS include a tip inside parentheses — 
   either (Tip: …) suggesting how the user could strengthen their position 
   or (Good move: …) if the user is doing something effective.

Keep the advice short and actionable (3–10 words).
Examples:
- (Tip: Emphasize value-added services.)
- (Good move: Showing flexibility builds rapport.)
- (Tip: Mention renewal discounts or exclusivity.)
"""


    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini not configured")

    try:
        response = gemini_model.generate_content(prompt)
        reply_text = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini negotiation call failed: {e}")

    # Save in session
    sessions[sid]["history"].append({"role": "user", "message": query})
    sessions[sid]["history"].append({"role": "assistant", "message": reply_text})

    return {
        "sessionId": sid,
        "response": reply_text,
        "retrieved_clauses": retrieved,
        "conversation_length": len(sessions[sid]["history"]),
    }

# ---------------- Create Important Points endpoint ----------------
class HighlightsInput(BaseModel):
    sessionId: Optional[str] = None
    conversation: str
    context: Optional[str] = None


@app.post("/create_highlights/")
def create_highlights(payload: HighlightsInput = Body(...)):
    """
    Summarize the negotiation conversation into key highlights, strategies, and takeaways.
    """

    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini not configured")

    conversation_text = payload.conversation.strip()
    if not conversation_text:
        raise HTTPException(status_code=400, detail="Conversation text cannot be empty")

    prompt = f"""
You are a legal negotiation summarizer AI.

Below is the conversation between a user and an AI contract negotiator.

Conversation:
\"\"\"{conversation_text}\"\"\"

Context:
{payload.context or "General negotiation"}

Task:
Summarize the most important points and strategic takeaways in clear bullet points.
Focus on:
- Key negotiation moves and offers made
- Strong arguments or leverage used
- Weak points that could be improved
- Suggested next steps or closing tactics

Return a plain text list (no JSON, no markdown).
Each point should be concise (1–2 lines).
"""

    try:
        response = gemini_model.generate_content(prompt)
        highlights_text = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini summary generation failed: {e}")

    sid = payload.sessionId or f"session_{uuid.uuid4().hex[:10]}"
    if sid not in sessions:
        sessions[sid] = {"highlights": highlights_text}
    else:
        sessions[sid]["highlights"] = highlights_text

    return {"sessionId": sid, "highlights": highlights_text}



# ---------------- Health ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": gemini_model is not None,
        "playbook_index_exists": os.path.exists(PLAYBOOK_INDEX_PATH) and os.path.exists(PLAYBOOK_META_PATH)
    }
