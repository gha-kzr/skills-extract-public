"""Dictionary extraction: find known skills (and their aliases) in an offer.

Usage: uv run dict_extract.py --offer offer.txt --dict data/skills.csv
"""
import argparse
import csv
import re

parser = argparse.ArgumentParser()
parser.add_argument("--offer", default="offer.txt")
parser.add_argument("--dict", default="data/skills.csv")
args = parser.parse_args()

text = open(args.offer, encoding="utf-8").read()

found = set()
with open(args.dict, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        names = [row["skill"]] + [a for a in (row["aliases"] or "").split("|") if a]
        for name in names:
            # whole-word, case-insensitive match
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
                found.add(row["skill"])

for skill in sorted(found):
    print(skill)
