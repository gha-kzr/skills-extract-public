"""NER extraction: a pretrained model tags skill spans in the offer.

The model (~1 GB) is downloaded from Hugging Face on first run, then cached.

Usage: uv run ner_extract.py --offer offer.txt
"""
import argparse

from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--offer", default="offer.txt")
args = parser.parse_args()

text = open(args.offer, encoding="utf-8").read()

# Multilingual model, tags "hard skills" (technologies, languages, platforms).
ner = pipeline(
    "token-classification",
    model="jjzha/escoxlmr_knowledge_extraction",
    aggregation_strategy="none",  # raw B/I/O tags, merged below
)

# The model has a 512-token window: run it line by line so nothing is cut.
# Tags are B (begin), I (inside), O (outside). Consecutive B/I tokens form a
# span, then it is grown to whole words (the model may start mid-word) and
# dropped if its average score is low.
skills = set()


def close(line, start, end, scores):
    if start is None or sum(scores) / len(scores) < 0.7:
        return
    while start > 0 and line[start - 1].isalnum():
        start -= 1
    while end < len(line) and line[end].isalnum():
        end += 1
    skills.add(line[start:end].strip())


for line in text.splitlines():
    start = end = None
    scores = []
    for tok in ner(line) + [{"entity": "O"}]:  # sentinel closes the last span
        if tok["entity"] == "B":
            close(line, start, end, scores)
            start, end, scores = tok["start"], tok["end"], [tok["score"]]
        elif tok["entity"] == "I" and start is not None:
            end = tok["end"]
            scores.append(tok["score"])
        else:
            close(line, start, end, scores)
            start = end = None
            scores = []

for skill in sorted(s for s in skills if s):
    print(skill)
