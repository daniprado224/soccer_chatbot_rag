"""Minimal FastAPI endpoint for querying the retrieval graph.

    uvicorn src.api:app --reload
    curl -X POST localhost:8000/query -H 'content-type: application/json' \\
        -d '{"question": "how many players are on a team?"}'
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from .graph import load_graph

app = FastAPI(title="Soccer Rules RAG")


@lru_cache(maxsize=1)
def _graph():
    return load_graph()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    cited_laws: list[str]
    answerable: bool
    retry_count: int


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = _graph().query(req.question)
    return QueryResponse(
        answer=result["answer"],
        cited_laws=result["cited_laws"],
        answerable=result["answerable"],
        retry_count=result["retry_count"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
