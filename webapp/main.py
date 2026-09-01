"""
Step 5: Web UI for the RAG pipeline.

A thin FastAPI layer over the existing scripts/retrieve.py + generate.py --
no pipeline logic is duplicated here, just exposed over HTTP. The Retriever
and Q/A link index are loaded once at startup (they're expensive: the
embedding model load alone takes a couple seconds) and reused across
requests.

Run with:
    GROQ_API_KEY=... uvicorn webapp.main:app --reload --port 8000
Then open http://localhost:8000
"""

import sys
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from retrieve import Retriever  # noqa: E402
from link_qa import load_links as load_qa_links  # noqa: E402
from generate import (  # noqa: E402
    generate_quiz, load_linked_chunk_index,
    GROQ_DEFAULT_MODEL, GEMINI_DEFAULT_MODEL,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="RAG Quiz Generator")

_state = {}


@app.on_event("startup")
def load_pipeline():
    t0 = time.time()
    print("Loading retriever (embedding model + vector store) ...")
    _state["retriever"] = Retriever()
    _state["qa_links"] = load_qa_links()
    _state["linked_index"] = load_linked_chunk_index(_state["qa_links"])
    print(f"Ready in {time.time() - t0:.1f}s "
          f"({len(_state['qa_links']) // 2} Q/A pairs indexed)")


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    n: int = Field(default=3, ge=1, le=8)
    provider: Literal["groq", "gemini"] = "groq"
    k: int = Field(default=6, ge=2, le=12)


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    model = GROQ_DEFAULT_MODEL if req.provider == "groq" else GEMINI_DEFAULT_MODEL
    try:
        result = generate_quiz(
            topic=req.topic, n=req.n, provider=req.provider, model=model, k=req.k,
            temperature=0.4, retriever=_state["retriever"],
            qa_links=_state["qa_links"], linked_index=_state["linked_index"],
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Model output failed validation: {e}")
    except RuntimeError as e:
        # Covers "no chunks retrieved", missing API key, and upstream API errors
        # (already retried with backoff inside generate.py before reaching here).
        msg = str(e)
        status = 400
        if "API_KEY is not set" in msg:
            status = 503
        elif "API" in msg and ("429" in msg or "5" in msg[:20]):
            status = 502
        raise HTTPException(status_code=status, detail=msg)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "qa_pairs": len(_state.get("qa_links", {})) // 2,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
