"""Minimal CLI for querying the retrieval graph.

    python -m src.cli "how many players are on a team?"
    python -m src.cli   # interactive loop
"""
from __future__ import annotations

import sys

from .graph import load_graph


def _print_result(result: dict) -> None:
    print("\n" + result["answer"])
    if result["cited_laws"]:
        print(f"[cited: Law {', '.join(result['cited_laws'])}]")
    if result["retry_count"]:
        print(f"[query was rewritten {result['retry_count']}x to find relevant passages]")


def main() -> None:
    graph = load_graph()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = graph.query(question)
        _print_result(result)
        return

    print("Soccer Rules RAG -- type a question, or 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        result = graph.query(question)
        _print_result(result)


if __name__ == "__main__":
    main()
