"""LangGraph retrieval/grading/answer flow.

State machine:

    retrieve --> grade --+-- (>=1 relevant OR already retried) --> answer
                          |
                          +-- (0 relevant AND not yet retried) --> rewrite_query --> retrieve

``answer`` always runs last and is the only node allowed to produce
user-facing text; if grading still finds nothing relevant after the retry,
it returns "I don't know" rather than letting the LLM free-associate from
its own training data.
"""
from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from .llm import DEFAULT_MODEL, GRADER_MODEL, call_llm, extract_json

load_dotenv()

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MIN_RELEVANT = int(os.getenv("RELEVANCE_MIN_RELEVANT", "1"))
MAX_RETRIES = 1


class RetrievedChunk(TypedDict):
    text: str
    law_number: str
    law_title: str
    section_title: str
    source_doc: str
    score: float
    relevant: bool | None


def _normalize_law_number(value: str) -> str:
    """Extract a bare law number from a citation, regardless of how the LLM
    formatted it ("Law 3", "Art. 3", "3", "law no. 3" all -> "3").

    Prompted output format isn't reliable across models -- Gemini in
    particular returned "Law 3" instead of the requested "3" during manual
    testing -- and downstream eval matching compares against bare numbers
    from qa_pairs.json, so normalize defensively rather than trust the
    prompt alone.
    """
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else str(value).strip()


class GraphState(TypedDict):
    original_question: str
    question: str
    retrieved: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    retry_count: int
    answer: str
    cited_laws: list[str]
    answerable: bool


class RetrievalGraph:
    """Wraps a Chroma retriever + LLM grading/generation into a LangGraph app.

    The LLM backend (Claude by default, Gemini as a free-tier alternative)
    is selected globally via src/llm.py -- this class just calls call_llm().
    """

    def __init__(self, vectorstore, top_k: int = TOP_K, generation_model: str = DEFAULT_MODEL,
                 grader_model: str = GRADER_MODEL):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.generation_model = generation_model
        self.grader_model = grader_model
        self.app = self._build_graph()

    # ---- nodes ----

    def _retrieve(self, state: GraphState) -> dict:
        results = self.vectorstore.similarity_search_with_score(state["question"], k=self.top_k)
        retrieved: list[RetrievedChunk] = []
        for doc, score in results:
            retrieved.append(
                RetrievedChunk(
                    text=doc.page_content,
                    law_number=doc.metadata.get("law_number", ""),
                    law_title=doc.metadata.get("law_title", ""),
                    section_title=doc.metadata.get("section_title", ""),
                    source_doc=doc.metadata.get("source_doc", ""),
                    score=float(score),
                    relevant=None,
                )
            )
        return {"retrieved": retrieved}

    def _grade(self, state: GraphState) -> dict:
        graded: list[RetrievedChunk] = []
        for chunk in state["retrieved"]:
            label = self._grade_one(state["original_question"], chunk)
            chunk = dict(chunk)
            chunk["relevant"] = label
            if label:
                graded.append(chunk)
        return {"relevant_chunks": graded}

    def _grade_one(self, question: str, chunk: RetrievedChunk) -> bool:
        system = (
            "You are a strict relevance grader for a rules-lookup RAG system. "
            "Given a question and one retrieved passage, answer whether the "
            "passage could help answer the question -- even partially. "
            'Respond with ONLY a JSON object: {"relevant": true} or {"relevant": false}.'
        )
        user = (
            f"Question: {question}\n\n"
            f"Passage (Law {chunk['law_number']} - {chunk['section_title']}):\n{chunk['text']}"
        )
        try:
            raw = call_llm(system, user, self.grader_model, max_tokens=20)
            return bool(extract_json(raw).get("relevant", False))
        except Exception:
            # Fail closed: an ungraded chunk is treated as not relevant
            # rather than silently passed through to generation.
            return False

    def _should_rewrite(self, state: GraphState) -> str:
        if len(state["relevant_chunks"]) >= MIN_RELEVANT:
            return "answer"
        if state["retry_count"] >= MAX_RETRIES:
            return "answer"
        return "rewrite"

    def _rewrite_query(self, state: GraphState) -> dict:
        system = (
            "You rewrite football (soccer) rules questions to improve retrieval "
            "against the IFAB Laws of the Game and related FIFA documents. "
            "Expand abbreviations, add likely official terminology, keep it a single "
            'question. Respond with ONLY JSON: {"rewritten_question": "..."}'
        )
        user = f"Original question: {state['question']}\nNo relevant passages were found for it."
        try:
            raw = call_llm(system, user, self.generation_model, max_tokens=200)
            rewritten = extract_json(raw).get("rewritten_question", state["question"])
        except Exception:
            rewritten = state["question"]
        return {"question": rewritten, "retry_count": state["retry_count"] + 1}

    def _answer(self, state: GraphState) -> dict:
        chunks = state["relevant_chunks"]
        if not chunks:
            return {
                "answer": (
                    "I don't know -- I couldn't find anything in the Laws of the Game "
                    "or the supplementary documents that answers this question."
                ),
                "cited_laws": [],
                "answerable": False,
            }

        context = "\n\n---\n\n".join(
            f"[Law {c['law_number']} - {c['section_title']} | {c['source_doc']}]\n{c['text']}"
            for c in chunks
        )
        system = (
            "You are a football (soccer) rules assistant. Answer ONLY using the "
            "provided context passages -- never from outside knowledge. Cite every "
            "law/article number you rely on inline, like '(Law 12)'. If the context "
            "does not actually answer the question, say you don't know instead of "
            "guessing. Respond with ONLY a JSON object: "
            '{"answer": "...", "cited_laws": ["12", "5"], "answerable": true|false}'
        )
        user = f"Context:\n{context}\n\nQuestion: {state['original_question']}"
        try:
            raw = call_llm(system, user, self.generation_model, max_tokens=800)
            parsed = extract_json(raw)
            return {
                "answer": parsed.get("answer", ""),
                "cited_laws": [_normalize_law_number(x) for x in parsed.get("cited_laws", [])],
                "answerable": bool(parsed.get("answerable", True)),
            }
        except Exception as e:
            return {
                "answer": f"Generation failed: {e}",
                "cited_laws": [],
                "answerable": False,
            }

    # ---- graph wiring ----

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite_query)
        graph.add_node("answer", self._answer)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self._should_rewrite, {"answer": "answer", "rewrite": "rewrite"})
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("answer", END)
        return graph.compile()

    def query(self, question: str) -> GraphState:
        initial: GraphState = {
            "original_question": question,
            "question": question,
            "retrieved": [],
            "relevant_chunks": [],
            "retry_count": 0,
            "answer": "",
            "cited_laws": [],
            "answerable": True,
        }
        return self.app.invoke(initial)


def load_graph(persist_dir: str | None = None, collection: str | None = None) -> RetrievalGraph:
    from langchain_chroma import Chroma

    from .embeddings import get_embeddings

    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection = collection or os.getenv("CHROMA_COLLECTION", "soccer_laws")
    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    return RetrievalGraph(store)
