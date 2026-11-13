"""NER extraction: a pretrained model tags skill spans in the offer.

The model (~1 GB) is downloaded from Hugging Face on first run, then cached.

Usage: uv run src/ner_extract.py [--offer data/offer.txt]
"""
from pathlib import Path

import typer

from lib import DATA, find_spans, whole_words

MIN_SCORE = 0.7  # drop spans the model is not confident about


def main(offer: Path = DATA / "offer.txt"):
    text = offer.read_text(encoding="utf-8")

    skills = set()
    # The model reads at most 512 tokens: go line by line so nothing is cut.
    for line in text.splitlines():
        for start, end, score in find_spans(line):
            if score >= MIN_SCORE:
                skills.add(whole_words(line, start, end))

    for skill in sorted(skills):
        print(skill)


if __name__ == "__main__":
    typer.run(main)
