"""NER extraction: a pretrained model tags skill spans in the offer.

The model (~1 GB) is downloaded from Hugging Face on first run, then cached.

Usage: uv run src/ner_extract.py [--offer data/offer.txt]
"""
from pathlib import Path

import typer
from transformers import pipeline

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_SCORE = 0.7  # drop spans the model is not confident about

# Multilingual model that tags "hard skills" (technologies, languages, platforms).
# It returns one tag per token: B (begins a skill), I (inside a skill), O (outside).
ner = pipeline(
    "token-classification",
    model="jjzha/escoxlmr_knowledge_extraction",
    aggregation_strategy="none",
)


def find_spans(line: str) -> list[tuple[int, int, float]]:
    """Group the tokens of a line into skills: (start, end, mean score) in characters.

    "Vous maîtrisez Apache Kafka et Python" is tagged  O O B I O B,
    which gives two spans: "Apache Kafka" and "Python".
    """
    spans = []
    current = None  # [start, end, scores] of the skill being read
    for token in ner(line):
        if token["entity"] == "B":  # a new skill starts: the previous one is over
            if current:
                spans.append(current)
            current = [token["start"], token["end"], [token["score"]]]
        elif token["entity"] == "I" and current:  # the skill continues
            current[1] = token["end"]
            current[2].append(token["score"])
        else:  # "O": the skill is over
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)
    return [(start, end, sum(scores) / len(scores)) for start, end, scores in spans]


def whole_words(line: str, start: int, end: int) -> str:
    """The model can start or stop in the middle of a word: widen to full words."""
    while start > 0 and line[start - 1].isalnum():
        start -= 1
    while end < len(line) and line[end].isalnum():
        end += 1
    return line[start:end].strip()


def extract(text: str) -> set[str]:
    skills = set()
    # The model reads at most 512 tokens: go line by line so nothing is cut.
    for line in text.splitlines():
        for start, end, score in find_spans(line):
            if score >= MIN_SCORE:
                skills.add(whole_words(line, start, end))
    return skills


def main(offer: Path = DATA / "offer.txt"):
    for skill in sorted(extract(offer.read_text(encoding="utf-8"))):
        print(skill)


if __name__ == "__main__":
    typer.run(main)
