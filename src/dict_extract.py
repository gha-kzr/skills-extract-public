"""Dictionary extraction: find the known skills of a list in an offer.

Usage: uv run src/dict_extract.py [--offer data/offer.txt] [--dictionary data/skills.txt]
"""
import re
from pathlib import Path

import typer

from lib import DATA


def main(
    offer: Path = DATA / "offer.txt",
    dictionary: Path = DATA / "skills.txt",  # one skill per line
):
    text = offer.read_text(encoding="utf-8")
    known_skills = [s.strip() for s in dictionary.read_text(encoding="utf-8").splitlines() if s.strip()]

    for skill in known_skills:
        # whole word, any case
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text, re.IGNORECASE):
            print(skill)


if __name__ == "__main__":
    typer.run(main)
