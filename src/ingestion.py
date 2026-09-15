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
MIN_CHUNK_CHARS = 15


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
    # Optional running header/footer pattern, e.g. IFAB's own
    # "Laws of the Game 2026/27 | Law 3 | The Players 61" printed on nearly
    # every content page. When present and it actually matches somewhere in
    # the document, it is used INSTEAD of unit_regex to assign each page's
    # law number -- the real per-Law chapter-opener graphics in this PDF
    # turned out to be rasterized title art with no extractable heading
    # text, so unit_regex alone only ever caught 2 of the 17 Laws. The
    # footer, by contrast, repeats on essentially every page. Must capture
    # (number, title) as groups 1 and 2.
    footer_regex: re.Pattern | None = None


# Matches e.g. "Law 3 – The Players", "Law 12: Fouls and Misconduct", "LAW 11 OFFSIDE"
DOC_PROFILES: dict[str, DocProfile] = {
    "laws_of_the_game": DocProfile(
        doc_type="laws_of_the_game",
        unit_label="Law",
        # [^()]*$ deliberately excludes lines like "Law 3 - The Players (p. 60)"
        # -- those are Topic Finder / index entries, not real section starts.
        unit_regex=re.compile(
            r"^\s*Law\s+(\d{1,2})\b[\s:.\-–—]*([^()]*)$", re.IGNORECASE
        ),
        footer_regex=re.compile(
            r"Laws of the Game\s+\d{4}/\d{2}\s*\|\s*Law\s+(\d{1,2})\s*\|\s*(.+?)\s+\d+\s*$",
            re.IGNORECASE,
        ),
    ),
    "disciplinary_code": DocProfile(
        doc_type="disciplinary_code",
        unit_label="Article",
        unit_regex=re.compile(
            r"^\s*Art(?:icle)?\.?\s+(\d{1,3})\b[\s:.\-–—]*([^()]*)$", re.IGNORECASE
        ),
    ),
    "var_protocol": DocProfile(
        doc_type="var_protocol",
        unit_label="Section",
        unit_regex=re.compile(
            r"^\s*Section\s+(\d{1,2})\b[\s:.\-–—]*([^()]*)$", re.IGNORECASE
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


def _find_column_split(words: list[dict], page_width: float) -> float | None:
    """Find an x-coordinate splitting the page into two side-by-side columns.

    Some "double pages" PDF exports lay out two facing book pages side by
    side on one physical PDF page. Grouping words into lines by y-position
    alone then interleaves the two pages' text. We detect the gutter as the
    single widest horizontal gap between words, restricted to the middle of
    the page so we don't pick up a normal paragraph's ragged right margin.
    Returns None for a genuinely single-column page.
    """
    if not words:
        return None
    xs = sorted(w["x0"] for w in words)
    best_gap, best_pos = 0.0, None
    for a, b in zip(xs, xs[1:]):
        gap = b - a
        mid = (a + b) / 2
        if gap > best_gap and page_width * 0.3 < mid < page_width * 0.7:
            best_gap, best_pos = gap, mid
    if best_pos is not None and best_gap > page_width * 0.03:
        return best_pos
    return None


_HEADING_FONT_MARKERS = ("bold", "black", "extrabold", "heavy", "semibold")


def _lines_from_words(
    words: list[dict],
    body_size: float,
    body_font: str,
    page_number: int,
    use_text_fallback: bool = False,
) -> list[Line]:
    """Group a flat word list (already restricted to one column) into visual lines.

    Heading detection is font-driven, by MAJORITY vote across the line's words
    (not "any word"): a line is a heading candidate if most of its words are
    in a distinct, bold-ish font from the page's own modal body font, or are
    meaningfully larger. The text-shape heuristic is a fallback of last
    resort, only used when a page has no font variation to key off at all
    (e.g. everything embedded under one font name) -- on a real, professionally
    typeset document like this one it is too eager (it flags ordinary
    sentences like "The competition rules must state:" as headings) so it
    must not be a primary signal once real font metadata is available.
    """
    grouped: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"] / 2) * 2
        grouped.setdefault(key, []).append(w)

    lines: list[Line] = []
    for key in sorted(grouped):
        line_words = sorted(grouped[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if not text:
            continue
        n = len(line_words)
        avg_size = sum(w["size"] for w in line_words) / n
        bold_frac = sum(
            1 for w in line_words if any(m in w["fontname"].lower() for m in _HEADING_FONT_MARKERS)
        ) / n
        other_font_frac = sum(1 for w in line_words if w["fontname"] != body_font) / n
        is_bold = bold_frac >= 0.6
        is_different_font = other_font_frac >= 0.6
        is_oversized = avg_size > body_size * 1.08
        heading = is_bold or is_different_font or is_oversized
        if use_text_fallback and not heading:
            heading = _looks_like_heading(text)
        # Folio numbers and lone bullet glyphs pick up the same emphasis
        # font as real headings but aren't section boundaries -- letting
        # them through would wrongly reset the current section on every
        # bullet point in a list.
        stripped = text.strip("••-*–— \t")
        if heading and (not stripped or stripped.isdigit()):
            heading = False
        lines.append(Line(text=text, page_number=page_number, is_heading=heading))
    return lines


def extract_lines_from_pdf(pdf_path: str | Path) -> list[Line]:
    """Extract text lines from a PDF, flagging heading candidates via font size/weight.

    Handles two-column "double pages" spreads (see ``_find_column_split``) by
    reading the left column top-to-bottom, then the right column top-to-bottom,
    for each physical PDF page.
    """
    import pdfplumber

    lines: list[Line] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size"])
            if not words:
                continue

            sizes = Counter(round(w["size"], 1) for w in words)
            body_size = sizes.most_common(1)[0][0]
            fontnames = Counter(w["fontname"] for w in words)
            body_font = fontnames.most_common(1)[0][0]
            # No font variation at all on this page -> font-based heading
            # detection has nothing to key off; fall back to text shape.
            use_text_fallback = len(fontnames) <= 1

            split_x = _find_column_split(words, page.width)
            if split_x is None:
                columns = [words]
            else:
                columns = [
                    [w for w in words if w["x0"] < split_x],
                    [w for w in words if w["x0"] >= split_x],
                ]

            for col_words in columns:
                lines.extend(
                    _lines_from_words(col_words, body_size, body_font, page_number, use_text_fallback)
                )
    return lines


def _index_like_pages(lines: list[Line], profile: DocProfile) -> set[int]:
    """Detect index / "Topic Finder" pages so their headers aren't treated as
    real section boundaries.

    A real chapter opener has exactly one "Law N" heading per page (or two,
    on facing pages of a title spread). A table-of-contents-style index page
    packs several DIFFERENT law numbers' headings onto one page -- that
    clustering is the signal we key off, since it's robust to whatever
    punctuation/typography distinguishes the two in a given edition.
    """
    by_page: dict[int, set[str]] = {}
    for line in lines:
        if not line.is_heading:
            continue
        m = profile.unit_regex.match(line.text)
        if m:
            by_page.setdefault(line.page_number, set()).add(m.group(1))
    return {page for page, laws in by_page.items() if len(laws) >= 3}


_FOOTER_FORWARD_FILL_MAX_GAP = 4  # observed real max gap between footer sightings is 2 pages


def _footer_law_by_page(
    lines: list[Line], profile: DocProfile, max_gap: int = _FOOTER_FORWARD_FILL_MAX_GAP
) -> dict[int, tuple[str, str]]:
    """Page -> (law_number, law_title), forward-filled across small gaps only.

    Direct footer sightings are ~1-2 pages apart throughout the real Laws
    1-17 content. A gap much larger than that means we've left the
    Law-numbered content entirely (e.g. this document's ~35-page appendix
    of "Additional instructions" after Law 17, which has no footer at all)
    -- forward-filling through a gap that large would mislabel all of it as
    whatever law came last. Pages beyond max_gap of the last direct sighting
    are left unmapped (dropped, like front matter) rather than mislabeled.
    """
    if not profile.footer_regex:
        return {}
    direct: dict[int, tuple[str, str]] = {}
    for line in lines:
        m = profile.footer_regex.search(line.text)
        if m:
            direct[line.page_number] = (m.group(1), m.group(2).strip())
    if not direct:
        return {}

    filled: dict[int, tuple[str, str]] = {}
    last_page, last_law = None, None
    for page in range(min(direct), max(direct) + 1):
        if page in direct:
            last_page, last_law = page, direct[page]
            filled[page] = last_law
        elif last_page is not None and page - last_page <= max_gap:
            filled[page] = last_law
        # else: gap too large -- leave this page unmapped
    return filled


def segment_into_chunks(
    lines: list[Line], profile: DocProfile, source_doc: str
) -> list[Chunk]:
    """Split lines into rule-numbered chunks. Pure function -- no PDF I/O."""
    chunks: list[Chunk] = []
    index_pages = _index_like_pages(lines, profile)

    footer_by_page = _footer_law_by_page(lines, profile)
    # Only trust the footer as the source of truth for top-level law
    # boundaries if it actually matched somewhere in THIS document -- e.g.
    # this profile might be reused for an edition that doesn't have it, or
    # for a doc_type where footer_regex isn't set at all.
    use_footer = bool(footer_by_page)
    footer_page_law: tuple[str, str] | None = None  # forward-filled as we scan

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
        if line.page_number in index_pages:
            # Topic Finder / index pages are page-reference listings, not
            # citable rule text -- skip them entirely rather than let their
            # heading-styled entries either reset the current law or leak
            # index text into whatever chunk was open.
            continue

        if use_footer:
            # Real chapter-opener headings in this document turned out to be
            # rasterized title art (no extractable text), so the running
            # footer -- present on nearly every content page -- is the
            # reliable signal for "which Law is this page" instead.
            # footer_by_page is already gap-limited-forward-filled per page;
            # a page absent from it (e.g. the trailing appendix) means "no
            # longer inside Law-numbered content", not "keep the last law".
            page_law = footer_by_page.get(line.page_number)
            if page_law != footer_page_law:
                flush()
                footer_page_law = page_law
                law_number, law_title = page_law if page_law else (None, None)
                section_title = None
                page_start = line.page_number
                page_end = line.page_number
                # Don't also append the footer line itself as body content below.
                if profile.footer_regex.search(line.text):
                    continue

            if profile.footer_regex.search(line.text):
                continue  # footer line on a page whose law didn't change -- still not body content
        else:
            # Fallback for doc types/editions with no reliable footer:
            # require is_heading too, not just a text match -- real
            # documents reference "Law 6" in prose all the time ("...as
            # Law 6 stipulates, competition rules must...") and only a
            # genuine heading-styled line marks an actual new section.
            unit_match = profile.unit_regex.match(line.text) if line.is_heading else None
            if unit_match:
                flush()
                law_number = unit_match.group(1)
                title = (
                    unit_match.group(2).strip()
                    if unit_match.lastindex and unit_match.lastindex >= 2
                    else ""
                )
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

    # Drop near-empty fragments: isolated diagram dimension labels, stray
    # bullet glyphs, and folio numbers that pick up a heading-like font in
    # the source PDF but carry no retrievable content of their own. The
    # real content they were adjacent to survives in a neighboring chunk.
    before = len(final)
    final = [c for c in final if len(c.text.strip()) >= MIN_CHUNK_CHARS]
    dropped_fragments = before - len(final)

    if dropped_front_matter:
        print(
            f"[ingestion] dropped {dropped_front_matter} line-group(s) before the "
            f"first '{profile.unit_label} N' heading in {source_doc} (title page / "
            "table of contents) -- not citable rule content.",
            file=sys.stderr,
        )
    if dropped_fragments:
        print(
            f"[ingestion] dropped {dropped_fragments} near-empty fragment chunk(s) "
            f"(< {MIN_CHUNK_CHARS} chars) in {source_doc} -- diagram labels/stray glyphs, "
            "not retrievable content.",
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
