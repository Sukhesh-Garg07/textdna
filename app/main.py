"""TextDNA 2.0 -- FastAPI service.

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://127.0.0.1:8000 for the web page, or POST JSON to
/analyze. All analysis runs locally on CPU; no text leaves the machine.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .analyzer import analyze

app = FastAPI(
    title="TextDNA",
    description="Multi-layer NLP text-intelligence engine: linguistic, "
    "semantic, argument-mining and cognitive analysis with a composite "
    "TextDNA Score.",
    version="2.0.0",
)

_TEMPLATE = Path(__file__).parent / "templates" / "index.html"


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


@app.post("/analyze")
def analyze_endpoint(payload: AnalyzeRequest) -> dict:
    if not payload.text.strip():
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="text must not be blank")
    return analyze(payload.text)
