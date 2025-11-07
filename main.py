from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import io
from pypdf import PdfReader
import re

# This allows CORS for your Next.js app to call this backend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CORS Middleware Setup ---
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

# --- Pydantic Models ---
class PDFUrl(BaseModel):
    pdf_url: str

class Clause(BaseModel):
    clause_type: str
    clause_text: str

# --- Helper Functions ---
def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extracts raw text from PDF content."""
    try:
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse PDF content.")

def extract_clauses_from_text(text: str) -> List[Clause]:
    """
    Uses a simple heuristic to split text into clauses.
    This can be significantly improved with better regex or NLP models.
    """
    # This regex splits the text by lines that look like numbered headings
    potential_clauses = re.split(r'\n\s*(?:\d+\.|\d+\.\d+|[a-z]\))\s+', text)
    
    clauses = []
    for clause_text in potential_clauses:
        cleaned_text = clause_text.strip()
        if len(cleaned_text) > 25: # Filter out short, irrelevant lines
            lines = cleaned_text.split('\n')
            clause_type = lines[0].strip()
            clause_body = '\n'.join(lines[1:]).strip()

            if not clause_type: # Fallback if first line is empty
                clause_type = "Uncategorized"
                clause_body = cleaned_text

            clauses.append(Clause(
                clause_type=clause_type,
                clause_text=clause_body
            ))
    return clauses

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/extract-clauses", response_model=List[Clause])
def extract_clauses(data: PDFUrl):
    """
    Receives a PDF URL, downloads it, parses the text,
    and extracts legal clauses.
    """
    print(f"Received URL to process: {data.pdf_url}")
    
    try:
        # 1. Download the PDF from the URL
        response = requests.get(data.pdf_url)
        response.raise_for_status()  # Raises an exception for bad status codes
        
        # 2. Extract text from the PDF content
        pdf_text = extract_text_from_pdf(response.content)
        
        # 3. Extract clauses from the text
        clauses = extract_clauses_from_text(pdf_text)
        
        if not clauses:
            raise HTTPException(status_code=404, detail="No clauses could be extracted from the document.")
            
        return clauses

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {e}")
    except Exception as e:
        # Catches errors from parsing or clause extraction
        raise HTTPException(status_code=500, detail=str(e))