"""Decision-model extraction: ask Jev (a System One model) about every 1-2 word n-gram.

The offer is sent once as the `state`; each n-gram is one yes/no question
("is X a technology, tool, technical concept or method?") answered with a
probability. We keep the n-grams above a threshold.

Reads TYPESAFE_API_KEY from the environment.

Usage: uv run --env-file .env src/jev_extract.py [--offer data/offer.txt] [--threshold 0.8]
"""
from pathlib import Path

import typer

from lib import DATA, ask_jev, generate_candidates


def main(offer: Path = DATA / "offer.txt", threshold: float = 0.8):
    text = offer.read_text(encoding="utf-8")

    candidates = generate_candidates(text)
    probabilities = ask_jev(text, candidates)

    for candidate, probability in zip(candidates, probabilities):
        if probability >= threshold:
            print(candidate)


if __name__ == "__main__":
    typer.run(main)
