# Soccer Rules RAG

A retrieval-augmented chatbot that answers questions about the IFAB Laws of the Game (soccer/football). It retrieves relevant rule passages, grades them for relevance before answering, cites the specific Law number(s) it used, and says "I don't know" instead of guessing when nothing relevant is found.

**Live demo:** https://soccer-chat-rag.vercel.app
**Eval report:** [eval_report.md](eval_report.md)

## Try it

Ask things like:
- "How many players are on each team?"
- "Is it a handball if the ball hits a natural arm position?"
- "What was the final score of the 2022 World Cup final?" (should refuse — out of scope)

Every answer shows the actual retrieval ranking it used (expand "Retrieval ranking" under the answer), and if it can't answer, it suggests a rephrasing that scores higher against the retriever.

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (offline)"]
        PDF["Source PDF\n(IFAB Laws of the Game)"]
        Parse["pdfplumber: extract text\n+ font-based heading detection"]
        Segment["Split into chunks by\nLaw number, then sub-heading"]
        Embed["Embed chunks"]
        Store[(Vector store\nlaw_number, section_title, source_doc)]
        PDF --> Parse --> Segment --> Embed --> Store
    end

    subgraph Query["Query time (LangGraph)"]
        Q[/"User question"/]
        Retrieve["retrieve\ntop-k similarity search"]
        Grade["grade\nLLM judges each chunk\nrelevant / not relevant"]
        Decide{">=1 relevant\nor already retried?"}
        Rewrite["rewrite_query\nLLM reformulates the question"]
        Answer["answer\nLLM generates answer\nciting Law N, or refuses"]
        Out[/"Answer + cited law numbers"/]

        Q --> Retrieve --> Grade --> Decide
        Decide -->|no| Rewrite --> Retrieve
        Decide -->|yes| Answer --> Out
    end

    Store -.-> Retrieve
```

## Run it locally

```bash
git clone https://github.com/daniprado224/soccer_chatbot_rag.git
cd soccer_chatbot_rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set an LLM backend — either `LLM_BACKEND=claude` with `ANTHROPIC_API_KEY`, or `LLM_BACKEND=gemini` with a free key from https://aistudio.google.com/apikey. Embeddings default to a local spaCy model (no key needed).

You'll need the IFAB Laws of the Game PDF — download it from https://www.theifab.com/laws-of-the-game-documents/ and place it in `data/raw/`. Then ingest it:

```bash
python -m src.ingestion --pdf "data/raw/<your file>.pdf" --doc-type laws_of_the_game --dump-chunks data/processed/chunks.json
```

Query it:

```bash
python -m src.cli "how many players are on a team?"
python -m src.cli                     # interactive
```

Run tests (no API key or PDF required):

```bash
pytest tests/ -v
```

## Repo layout

```
data/
  raw/            # source PDF (gitignored — copyrighted, download it yourself)
  processed/      # chunks.json + embeddings.npy
  eval/           # ground-truth QA pairs
src/
  ingestion.py    # PDF -> section-aware chunks -> vector store
  embeddings.py   # embedding backend (spacy | openai)
  llm.py          # LLM backend (claude | gemini)
  graph.py        # LangGraph retrieve/grade/rewrite/answer flow
  memory_vectorstore.py  # in-memory vector store used by the deployed app
  quota_store.py  # shared usage tracker (Upstash Redis) for the deployed app
  query_diagnostics.py   # retrieval ranking + phrasing suggestions
  cli.py, api.py, eval.py
api/query.py      # Vercel serverless function
public/index.html # web frontend
tests/
```

## Evaluation

```bash
python -m src.eval --qa data/eval/qa_pairs.json --top-k 5 --out eval_report.md --json-out eval_results.json
```

Retrieval metrics (precision@k / recall@k) are measured before the LLM grader runs, so retrieval quality is scored separately from generation quality. Every failing question is bucketed as `retrieval` (right law never retrieved), `generation` (retrieved but not cited), `over_refusal`, or `hallucination` — see `eval_report.md` for the full per-question breakdown.

26 ground-truth questions: 10 easy factual lookups, 10 ambiguous/edge-case questions, 6 out-of-corpus questions that should be refused.

**Results** (Gemini 3.1 Flash Lite, 144-chunk corpus):

| | n | Precision@k | Recall@k |
|---|---|---|---|
| Retrieval (overall) | 20 | 0.19 | 0.55 |

| | n | Citation overlap | Answerability correct |
|---|---|---|---|
| Generation (overall) | 26 | 0.35 | 0.46 |

- 0 hallucinations — all 6 out-of-corpus questions correctly refused.
- `over_refusal` is the dominant failure mode (14/20 answerable questions). Several of these had the correct chunk in the top-5 retrieval results but the grader still rejected it — pointing at grader strictness, not just weak retrieval, as the main thing to tune next.
- Precision@k of 0.19 reflects the embedding model choice (see below).

## Design choices

**Embeddings**: local spaCy word vectors by default, no API cost. Swappable to OpenAI via `EMBEDDING_BACKEND=openai`. Averaged word vectors are noticeably weaker than a real sentence embedding model at capturing meaning — this shows up directly in the retrieval numbers above and is the main lever for improving this project further.

**LLM backend**: Claude by default (`LLM_BACKEND=claude`), Gemini as a free alternative (`LLM_BACKEND=gemini`). Anthropic's API has no free tier — Gemini's free tier via Google AI Studio does, at the cost of tighter rate limits (as low as 15 requests/minute on some models) and lower generation quality than Claude. Both backends share one interface (`src/llm.py`), so switching is a one-line env var change; the eval numbers above are specific to whichever backend produced them.

**Deployment**: the web app uses an in-memory vector store instead of the local Chroma index, since Vercel's serverless functions don't have persistent disk. With ~150 chunks this is fast and simple — no hosted vector DB needed. A shared usage tracker (Upstash Redis) shows real-time availability across all visitors, since everyone hits the same free-tier API key.

## Known limitations

- Weak retrieval on general/short questions — averaged word vectors reward vocabulary overlap over meaning, so a question phrased differently from the rulebook's own wording can miss the right chunk entirely.
- Only the main Laws of the Game PDF is ingested. The FIFA Disciplinary Code and VAR Protocol are wired into `src/ingestion.py` (`disciplinary_code`, `var_protocol` doc types) but not yet supplied.
- A ~35-page appendix after Law 17 in this edition isn't ingested (no reliable footer/heading marker to segment it by).
