# Soccer Rules RAG

A retrieval-augmented chatbot that answers questions about the IFAB Laws of
the Game (soccer/football), with a LangGraph agent that grades retrieved
passages for relevance and rewrites the query once before giving up, and an
eval harness that reports retrieval and generation accuracy **separately**.

## Status / honesty notes (read this first)

This repo was built in a network-restricted sandbox. Some things that would
normally be "just run it" required explicit tradeoffs — documented here
instead of glossed over:

- **Embeddings backend**: the spec called for OpenAI `text-embedding-3-small`,
  with local-model as a fallback if I wanted to avoid API cost. Neither
  `api.openai.com` nor `huggingface.co` (where most local sentence-embedding
  models live) were reachable from the build sandbox. `github.com` release
  assets were reachable, so the default backend is **spaCy's
  `en_core_web_md`** — averaged 300-dim static word vectors, installed as a
  plain pip wheel with no HF/OpenAI dependency. This is measurably weaker
  than a real sentence-transformer or OpenAI embedding at capturing
  paraphrase/semantic similarity (it has no attention mechanism or
  sentence-level training), which shows up directly in the retrieval
  precision/recall numbers below. `OpenAIEmbeddingsBackend` is implemented
  and wired behind the same interface for when you run this somewhere with
  API access — see [Swapping the embedding backend](#swapping-the-embedding-backend).
- **Generation model**: the spec named `claude-sonnet-4-6`, which is not a
  real model ID. `CLAUDE_MODEL` defaults to `claude-sonnet-5`; change it in
  `.env` to whatever's current when you run this.
- **Source PDFs**: `theifab.com` and `fifa.com` were also blocked from the
  build sandbox, so ingestion could not download them automatically. You
  supply the PDFs (see [Data](#data)).
- **Eval numbers below are only real once you've run `src/eval.py` yourself**
  against the actual Laws of the Game PDF and a valid `ANTHROPIC_API_KEY`.
  Anything still marked "pending" is not a completed feature.

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (src/ingestion.py, offline)"]
        PDF["Source PDFs\n(Laws of the Game,\nDisciplinary Code, VAR Protocol)"]
        Parse["pdfplumber: extract lines\n+ font-based heading detection"]
        Segment["segment_into_chunks():\nsplit on Law/Article/Section N,\nthen on sub-headings"]
        Embed["Embed chunks\n(spaCy word vectors, or OpenAI)"]
        Chroma[(Chroma vector store\nmetadata: law_number,\nsection_title, source_doc)]
        PDF --> Parse --> Segment --> Embed --> Chroma
    end

    subgraph Query["Query time (src/graph.py, LangGraph)"]
        Q[/"User question"/]
        Retrieve["retrieve\ntop-k similarity search"]
        Grade["grade\nClaude judges each chunk\nrelevant / not relevant"]
        Decide{">=1 relevant\nOR already retried?"}
        Rewrite["rewrite_query\nClaude reformulates the question"]
        Answer["answer\nClaude generates answer\nciting Law N, or refuses"]
        Out[/"Answer + cited law numbers"/]

        Q --> Retrieve --> Grade --> Decide
        Decide -->|no, retry_count=0| Rewrite --> Retrieve
        Decide -->|yes| Answer --> Out
    end

    Chroma -.-> Retrieve
```

## Repo layout

```
data/
  raw/            # source PDFs you provide (gitignored -- copyrighted third-party docs)
  processed/      # optional dumped chunks (--dump-chunks)
  eval/qa_pairs.json  # ground-truth eval set
src/
  ingestion.py    # PDF -> section-aware chunks -> Chroma
  embeddings.py   # pluggable embedding backends (spacy | openai)
  graph.py        # LangGraph retrieve/grade/rewrite/answer state machine
  cli.py          # interactive / one-shot CLI
  api.py          # minimal FastAPI endpoint
  eval.py         # retrieval + generation metrics, markdown report
tests/            # unit tests -- run without any API key or source PDF
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY at minimum
```

`en_core_web_md` is pinned as a direct wheel URL in `requirements.txt`, so
`pip install -r requirements.txt` installs it automatically. If that URL
ever moves, fall back to `python -m spacy download en_core_web_md`.

Run the unit tests (no API key or PDF required — they cover chunking logic,
the LangGraph state machine, and the eval metrics/report with synthetic
fixtures and a mocked Claude client):

```bash
pytest tests/ -v
```

## Data

Place these in `data/raw/` (not tracked in git — see `.gitignore` note):

1. **IFAB Laws of the Game** (current edition), from
   https://www.theifab.com/laws-of-the-game-documents/
2. **FIFA Disciplinary Code** (excerpts or full), from FIFA's publications
3. **VAR Protocol**, from IFAB/FIFA's VAR documentation

Only #1 is required to run the core pipeline; #2 and #3 are the "2-3
supplementary docs" from the spec and are wired into `src/ingestion.py`
via the `disciplinary_code` and `var_protocol` doc profiles, but the two
eval questions that depend on them (`s01`, `s02` in `qa_pairs.json`) are
kept in a separate `supplementary_pending` bucket with `expected_law_numbers:
null` until those documents are actually supplied and ingested — see the
note at the top of that file. Do not treat those two as scored until then.

## Ingestion

```bash
python -m src.ingestion \
  --pdf data/raw/laws_of_the_game_2025_26.pdf --doc-type laws_of_the_game \
  --pdf data/raw/fifa_disciplinary_code.pdf   --doc-type disciplinary_code \
  --pdf data/raw/var_protocol.pdf             --doc-type var_protocol \
  --dump-chunks data/processed/chunks.json
```

Each `--doc-type` selects a `DocProfile` (a regex for that document's own
rule-numbering convention: `Law N` / `Article N` / `Section N`) used to
split the document — see `DOC_PROFILES` in `src/ingestion.py`. The
`--dump-chunks` output is worth eyeballing after a real ingestion run: the
heading-detection heuristic (bold/oversized font, with a text-shape
fallback) was written and unit-tested against synthetic fixtures, not
against the real PDF's actual font metrics, so it may need a pass of
tuning once you see how the real document's fonts come through
`pdfplumber`.

## Querying

```bash
python -m src.cli "how many players are on a team?"
python -m src.cli                     # interactive

uvicorn src.api:app --reload          # POST /query {"question": "..."}
```

## Swapping the embedding backend

Set in `.env`:

```
EMBEDDING_BACKEND=openai
OPENAI_API_KEY=sk-...
```

and `pip install langchain-openai`. Nothing else changes — `graph.py` and
`ingestion.py` only depend on the LangChain `Embeddings` interface. Note:
switching backends means re-running ingestion (embedding spaces aren't
compatible across backends).

## Evaluation

```bash
python -m src.eval --qa data/eval/qa_pairs.json --top-k 5 \
  --out eval_report.md --json-out eval_results.json
```

This is the part the spec said matters most, so here's exactly what it
measures and why:

- **Retrieval metrics** (precision@k / recall@k against ground-truth Law
  numbers) are computed from the raw top-k similarity search, *before* the
  relevance grader runs — this isolates embedding/retrieval quality from
  anything the LLM does downstream.
- **Generation metrics** (citation overlap/exact match, answerability
  correctness) are computed from the full graph's final output, *after*
  grading, retry, and generation.
- **Failure attribution**: every question that doesn't fully pass is
  bucketed as `retrieval` (the right law was never in the top-k),
  `generation` (it was retrieved but not cited correctly),
  `over_refusal` (an answerable question got refused), or
  `hallucination` (an out-of-corpus question got answered instead of
  refused) — so you can point at *which stage* broke, not just that the
  final answer was wrong.
- The eval set (`data/eval/qa_pairs.json`) has 26 scored questions: 10 easy
  factual lookups, 10 ambiguous/edge-case interpretation questions
  (handball nuance, offside edge cases, VAR authority limits), and 6
  deliberately out-of-corpus questions to test refusal instead of
  hallucination. 2 more are pending supplementary-doc ingestion (see
  [Data](#data)).

### Results

**Not yet run.** This requires the real Laws of the Game PDF (in progress —
being supplied separately) and a valid `ANTHROPIC_API_KEY`. Run the command
above and paste the generated `eval_report.md` contents here — do not fill
in numbers by hand. `git diff` after a real run will show exactly the
table below being replaced with actual measured precision@k, recall@k,
citation accuracy, and the failure-mode breakdown.

<!-- EVAL_REPORT_START -->
_pending real ingestion + eval run_
<!-- EVAL_REPORT_END -->

## Known limitations

- spaCy averaged-word-vector embeddings will noticeably underperform a real
  sentence embedding model on paraphrased or compositional questions (e.g.
  "the guy who came off the bench" vs "substitute") — the eval report's
  retrieval precision/recall numbers are the honest measurement of that
  gap once run, not a guess.
- The section-heading detection in `ingestion.py` is heuristic
  (bold/oversized font, or a text-shape fallback) and was validated against
  synthetic fixtures in `tests/test_ingestion.py`, not the real PDF —
  expect to tune `_looks_like_heading` and the bold/size thresholds after
  a first real ingestion run.
- The relevance grader and query rewriter are themselves LLM calls, so
  their own error rate (mis-grading a relevant chunk as irrelevant, or a
  bad rewrite) is a source of noise in the generation metrics that this
  eval harness attributes to "generation," not "retrieval" — that's a
  modeling choice, not a hidden bug.
