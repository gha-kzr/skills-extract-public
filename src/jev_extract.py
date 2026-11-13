"""Decision-model extraction: ask Jev (a System One model) about every 1-2 word n-gram.

The offer is sent once as the `state`; each n-gram is one yes/no question
("is X a technology, tool, technical concept or method?") answered with a
probability. We keep the n-grams above a threshold.

Reads TYPESAFE_API_KEY from the environment.

Usage: uv run --env-file .env src/jev_extract.py [--offer data/offer.txt] [--threshold 0.8]
"""
import json
import os
import re
import urllib.request
from pathlib import Path

import typer

DATA = Path(__file__).resolve().parent.parent / "data"

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Written once in the state, so each question stays short.
DEFINITION = (
    "We list the skills of one job offer: named technologies (languages, "
    "frameworks, tools, platforms, cloud services, databases), technical "
    "concepts (cloud, API, security, monitoring...) and methods (Agile, "
    "DevOps, CI/CD, GitOps...). Soft skills, spoken languages, degrees, "
    "years of experience, job titles and company names are not skills."
)
QUESTION = 'Is "{candidate}" a technology, tool, technical concept or method?'

# A skill never starts or ends with one of these ("avec Terraform", "ArgoCD sur").
FUNCTION_WORDS = set(
    "le la les l un une des de d du au aux et ou en dans sur sous par pour avec sans "
    "via notre nos votre vos ce cette ces que qui à a the an of and or in on for with "
    "to by from at as is are our your you we".split()
)


def tokenize(text: str) -> list[str]:
    """Split the offer into words. "Prometheus/Grafana" gives two words,
    while c++, c#, node.js and ci/cd stay single words."""
    text = re.sub(r"(ci/cd|c\+\+|c#)|[/:;,()\[\]|]", lambda m: m.group(1) or " ", text, flags=re.I)
    return re.findall(r"[\w+#]+(?:[./][\w+#]+)*", text)


def generate_candidates(text: str, max_words: int = 2) -> list[str]:
    """Every n-gram of 1 to `max_words` words, without those that start or end
    with a function word. Duplicates are removed (case-insensitive)."""
    words = tokenize(text)
    candidates = {}
    for n in range(1, max_words + 1):
        for i in range(len(words) - n + 1):
            ngram = words[i : i + n]
            if ngram[0].lower() in FUNCTION_WORDS or ngram[-1].lower() in FUNCTION_WORDS:
                continue
            candidates.setdefault(" ".join(ngram).lower(), " ".join(ngram))
    return list(candidates.values())


def ask_jev(text: str, candidates: list[str]) -> list[float]:
    """One request: the offer once, one yes/no question per candidate.
    Returns the probability of "yes" of each candidate."""
    payload = {
        "model": MODEL,
        "state": f"{DEFINITION}\n\nJOB OFFER:\n{text}",
        "questions": {
            f"q{i}": {"type": "noul", "instructions": QUESTION.format(candidate=c)}
            for i, c in enumerate(candidates)
        },
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
            "User-Agent": "skills-extract-demo/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        answers = json.load(response)["answers"]
    return [answers[f"q{i}"]["noul"] for i in range(len(candidates))]


def extract(text: str, threshold: float) -> list[str]:
    candidates = generate_candidates(text)
    probabilities = ask_jev(text, candidates)
    return [c for c, p in zip(candidates, probabilities) if p >= threshold]


def main(offer: Path = DATA / "offer.txt", threshold: float = 0.8):
    for skill in extract(offer.read_text(encoding="utf-8"), threshold):
        print(skill)


if __name__ == "__main__":
    typer.run(main)
