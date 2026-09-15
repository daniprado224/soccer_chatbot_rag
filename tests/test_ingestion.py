"""Unit tests for the pure segmentation logic in src/ingestion.py.

These use synthetic Line fixtures instead of a real PDF, so they run
without any source document and without network access.
"""
from src.ingestion import DOC_PROFILES, Line, segment_into_chunks


def test_segments_by_law_number_and_title():
    profile = DOC_PROFILES["laws_of_the_game"]
    lines = [
        Line("Law 3 – The Players", page_number=10, is_heading=True),
        Line("Number of Players", page_number=10, is_heading=True),
        Line("A match is played by two teams, each with a maximum of eleven players.", page_number=10),
        Line("Substitution procedure", page_number=11, is_heading=True),
        Line("A substitute may only enter the field of play at the halfway line.", page_number=11),
        Line("Law 4 – The Players' Equipment", page_number=12, is_heading=True),
        Line("Safety", page_number=12, is_heading=True),
        Line("A player must not use equipment or wear anything dangerous.", page_number=12),
    ]

    chunks = segment_into_chunks(lines, profile, source_doc="laws_of_the_game_2025_26.pdf")

    assert [c.law_number for c in chunks] == ["3", "3", "4"]
    assert chunks[0].law_title == "The Players"
    assert chunks[0].section_title == "Number of Players"
    assert "eleven players" in chunks[0].text
    assert chunks[1].section_title == "Substitution procedure"
    assert chunks[2].law_number == "4"
    assert chunks[2].law_title == "The Players' Equipment"
    assert chunks[2].section_title == "Safety"


def test_drops_front_matter_before_first_unit_heading():
    profile = DOC_PROFILES["laws_of_the_game"]
    lines = [
        Line("Laws of the Game 2025/26", page_number=1, is_heading=True),
        Line("Table of Contents", page_number=2, is_heading=True),
        Line("Law 1 .......... 1", page_number=2),
        Line("Law 1 – The Field of Play", page_number=5, is_heading=True),
        Line("The field of play must be rectangular.", page_number=5),
    ]

    chunks = segment_into_chunks(lines, profile, source_doc="laws.pdf")

    assert len(chunks) == 1
    assert chunks[0].law_number == "1"
    assert "rectangular" in chunks[0].text


def test_article_profile_for_disciplinary_code():
    profile = DOC_PROFILES["disciplinary_code"]
    lines = [
        Line("Article 15 Serious foul play", page_number=20, is_heading=True),
        Line("A player who commits serious foul play is sanctioned with a red card.", page_number=20),
    ]

    chunks = segment_into_chunks(lines, profile, source_doc="disciplinary_code.pdf")

    assert len(chunks) == 1
    assert chunks[0].law_number == "15"
    assert chunks[0].doc_type == "disciplinary_code"


def test_oversized_chunk_is_soft_split():
    profile = DOC_PROFILES["laws_of_the_game"]
    profile.max_chunk_chars = 50  # force a split for this test
    long_sentence = "word " * 40
    lines = [
        Line("Law 12 – Fouls and Misconduct", page_number=1, is_heading=True),
        Line(long_sentence, page_number=1),
    ]

    chunks = segment_into_chunks(lines, profile, source_doc="laws.pdf")

    assert len(chunks) > 1
    assert all(c.law_number == "12" for c in chunks)
    profile.max_chunk_chars = 2500  # restore default for other tests
