"""Section-aware ingestion: PDF -> rule-numbered chunks -> Chroma.

Chunking is driven by each document's own rule-numbering structure (Law N
in the Laws of the Game, Article N in the disciplinary code, Section N in
the VAR protocol) rather than a fixed token window. Every chunk carries
its rule number and heading as metadata so the generation node can cite
"Law 12" instead of "chunk 7".

Two layers, deliberately separated so the segmentation logic is unit
testable without a real PDF:

1. ``extract_lines_from_pdf`` -- pdfplumber I/O. Groups words into lines
   and flags each line as bold/oversized (a heading candidate) by
   comparing its font to the page's modal body font.
2. ``segment_into_chunks`` -- pure function over ``Line`` objects. No PDF
   dependency, so tests/test_ingestion.py exercises it directly with
   synthetic fixtures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Line:
    text: str
    page_number: int
    is_heading: bool = False


@dataclass
class Chunk:
    text: str
    source_doc: str
    doc_type: str
    law_number: str | None
    law_title: str | None
    section_title: str | None
    page_start: int
    page_end: int

    def metadata(self) -> dict:
        return {
            "source_doc": self.source_doc,
            "doc_type": self.doc_type,
            "law_number": self.law_number or "",
            "law_title": self.law_title or "",
            "section_title": self.section_title or "",
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass
class DocProfile:
    doc_type: str
    unit_label: str  # "Law" | "Article" | "Section"
    unit_regex: re.Pattern
    max_chunk_chars: int = 2500


# Matches e.g. "Law 3 – The Players", "Law 12: Fouls and Misconduct", "LAW 11 OFFSIDE"
DOC_PROFILES: dict[str, DocProfile] = {
    "laws_of_the_game": DocProfile(
        doc_type="laws_of_the_game",
        unit_label="Law",
        unit_regex=re.compile(
            r"^\s*Law\s+(\d{1,2})\b[\s:.\-–—]*(.*)$", re.IGNORECASE
        ),
    ),
    "disciplinary_code": DocProfile(
        doc_type="disciplinary_code",
        unit_label="Article",
        unit_regex=re.compile(
            r"^\s*Art(?:icle)?\.?\s+(\d{1,3})\b[\s:.\-–—]*(.*)$", re.IGNORECASE
        ),
    ),
    "var_protocol": DocProfile(
        doc_type="var_protocol",
        unit_label="Section",
        unit_regex=re.compile(
            r"^\s*Section\s+(\d{1,2})\b[\s:.\-–—]*(.*)$", re.IGNORECASE
        ),
    ),
}


def _looks_like_heading(text: str) -> bool:
    """Heuristic fallback used only when font metadata is unavailable/unreliable."""
    text = text.strip()
    if not text or len(text) > 90:
        return False
    if text.endswith((".", ",", ";")):
        return False
    words = text.split()
    if not words:
        return False
    # Titles are short, don't read as a full sentence, and are usually
    # Title Case or ALL CAPS rather than lowercase-led prose.
    return text[0].isupper() and len(words) <= 12


def extract_lines_from_pdf(pdf_path: str | Path) -> list[Line]:
    """Extract text lines from a PDF, flagging heading candidates via font size/weight."""
    import pdfplumber

    lines: list[Line] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"])
            if not words:
                continue

            # Modal font size on the page approximates body text size.
            sizes = Counter(round(w["size"], 1) for w in words)
            body_size = sizes.most_common(1)[0][0]

            # Group words into visual lines by rounded vertical position.
            grouped: dict[int, list[dict]] = {}
            for w in words:
                key = round(w["top"] / 2) * 2
                grouped.setdefault(key, []).append(w)

            for key in sorted(grouped):
                line_words = sorted(grouped[key], key=lambda w: w["x0"])
                text = " ".join(w["text"] for w in line_words).strip()
                if not text:
                    continue
                avg_size = sum(w["size"] for w in line_words) / len(line_words)
                is_bold = any("bold" in w["fontname"].lower() for w in line_words)
                is_oversized = avg_size > body_size * 1.08
                heading = is_bold or is_oversized or _looks_like_heading(text)
                lines.append(Line(text=text, page_number=page_number, is_heading=heading))
    return lines


def segment_into_chunks(
    lines: list[Line], profile: DocProfile, source_doc: str
) -> list[Chunk]:
    """Split lines into rule-numbered chunks. Pure function -- no PDF I/O."""
    chunks: list[Chunk] = []

    law_number: str | None = None
    law_title: str | None = None
    section_title: str | None = None
    buffer: list[str] = []
    page_start: int | None = None
    page_end: int | None = None
    dropped_front_matter = 0

    def flush():
        nonlocal buffer, page_start, page_end
        text = " ".join(buffer).strip()
        buffer = []
        if not text:
            return
        if law_number is None:
            nonlocal dropped_front_matter
            dropped_front_matter += 1
            return
        chunks.append(
            Chunk(
                text=text,
                source_doc=source_doc,
                doc_type=profile.doc_type,
                law_number=law_number,
                law_title=law_title,
                section_title=section_title,
                page_start=page_start or 0,
                page_end=page_end or page_start or 0,
            )
        )

    for line in lines:
        unit_match = profile.unit_regex.match(line.text)
        if unit_match:
            flush()
            law_number = unit_match.group(1)
            title = unit_match.group(2).strip() if unit_match.lastindex and unit_match.lastindex >= 2 else ""
            law_title = title or None
            section_title = None
            page_start = line.page_number
            page_end = line.page_number
            continue

        if line.is_heading and law_number is not None:
            flush()
            section_title = line.text.strip()
            page_start = line.page_number
            page_end = line.page_number
            continue

        if page_start is None:
            page_start = line.page_number
        page_end = line.page_number
        buffer.append(line.text)

    flush()

    # Soft-split any oversized chunk on paragraph-ish boundaries so a
    # single dense Law section doesn't dominate the retriever's top-k.
    final: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= profile.max_chunk_chars:
            final.append(c)
            continue
        words = c.text.split()
        part, part_len, parts = [], 0, []
        for w in words:
            part.append(w)
            part_len += len(w) + 1
            if part_len >= profile.max_chunk_chars:
                parts.append(" ".join(part))
                part, part_len = [], 0
        if part:
            parts.append(" ".join(part))
        for p in parts:
            final.append(
                Chunk(
                    text=p,
                    source_doc=c.source_doc,
                    doc_type=c.doc_type,
                    law_number=c.law_number,
                    law_title=c.law_title,
                    section_title=c.section_title,
                    page_start=c.page_start,
                    page_end=c.page_end,
                )
            )

    if dropped_front_matter:
        print(
            f"[ingestion] dropped {dropped_front_matter} line-group(s) before the "
            f"first '{profile.unit_label} N' heading in {source_doc} (title page / "
            "table of contents) -- not citable rule content.",
            file=sys.stderr,
        )
    return final


def ingest_pdf(pdf_path: str | Path, doc_type: str, source_doc: str | None = None) -> list[Chunk]:
    if doc_type not in DOC_PROFILES:
        raise ValueError(f"Unknown doc_type {doc_type!r}; expected one of {list(DOC_PROFILES)}")
    profile = DOC_PROFILES[doc_type]
    source_doc = source_doc or Path(pdf_path).name
    lines = extract_lines_from_pdf(pdf_path)
    chunks = segment_into_chunks(lines, profile, source_doc)
    return chunks


def build_vectorstore(chunks: list[Chunk], persist_dir: str, collection_name: str):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from .embeddings import get_embeddings

    docs = [Document(page_content=c.text, metadata=c.metadata()) for c in chunks]
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if docs:
        ids = [f"{d.metadata['source_doc']}::{i}" for i, d in enumerate(docs)]
        store.add_documents(docs, ids=ids)
    return store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="append", required=True, help="Path to a source PDF (repeatable)")
    parser.add_argument(
        "--doc-type",
        action="append",
        required=True,
        choices=list(DOC_PROFILES),
        help="Doc profile for the matching --pdf (repeatable, same order)",
    )
    parser.add_argument("--source-doc", action="append", default=None, help="Display name override (repeatable)")
    parser.add_argument("--persist", default=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "soccer_laws"))
    parser.add_argument("--dump-chunks", default=None, help="Optional path to also dump chunks as JSON")
    args = parser.parse_args()

    if len(args.pdf) != len(args.doc_type):
        parser.error("--pdf and --doc-type must be given the same number of times, in matching order")
    names = args.source_doc or [None] * len(args.pdf)
    if len(names) != len(args.pdf):
        parser.error("--source-doc, if given, must match --pdf count")

    all_chunks: list[Chunk] = []
    for pdf_path, doc_type, name in zip(args.pdf, args.doc_type, names):
        print(f"[ingestion] parsing {pdf_path} as {doc_type} ...")
        chunks = ingest_pdf(pdf_path, doc_type, name)
        print(f"[ingestion]   -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if args.dump_chunks:
        Path(args.dump_chunks).write_text(
            json.dumps([{"text": c.text, **c.metadata()} for c in all_chunks], indent=2)
        )
        print(f"[ingestion] dumped chunks to {args.dump_chunks}")

    print(f"[ingestion] embedding {len(all_chunks)} chunks into Chroma at {args.persist} ...")
    build_vectorstore(all_chunks, args.persist, args.collection)
    print("[ingestion] done.")


if __name__ == "__main__":
    main()
