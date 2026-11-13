"""Dictionary extraction: find the known skills of a list in an offer.

Usage: uv run src/dict_extract.py [--offer data/offer.txt] [--dictionary data/skills.txt]
"""
import re
from pathlib import Path

import typer

DATA = Path(__file__).resolve().parent.parent / "data"


def extract(text: str, dictionary: list[str]) -> list[str]:
    """The skills of `dictionary` written in `text` (whole word, any case)."""
    return [
        skill
        for skill in dictionary
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text, re.IGNORECASE)
    ]


def main(
    offer: Path = DATA / "offer.txt",
    dictionary: Path = DATA / "skills.txt",  # one skill per line
):
    text = offer.read_text(encoding="utf-8")
    skills = [line.strip() for line in dictionary.read_text(encoding="utf-8").splitlines()]
    for skill in extract(text, [s for s in skills if s]):
        print(skill)


if __name__ == "__main__":
    typer.run(main)
