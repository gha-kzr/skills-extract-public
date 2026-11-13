"""Helpers shared by the extraction scripts (the scripts only keep their `main`).

Heavy imports (transformers) are done inside the functions that need them,
so running the dictionary script does not load the NER model libraries.
"""
import json
import os
import re
import urllib.request
from functools import cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------- NER
@cache
def _ner_model():
    """Multilingual model that tags "hard skills" (technologies, languages,
    platforms). Downloaded (~1 GB) on first use. It returns one tag per token:
    B (begins a skill), I (inside a skill), O (outside)."""
    from transformers import pipeline

    return pipeline(
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
    for token in _ner_model()(line):
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


# ---------------------------------------------------------------- Jev
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

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
        "model": JEV_MODEL,
        "state": f"{DEFINITION}\n\nJOB OFFER:\n{text}",
        "questions": {
            f"q{i}": {"type": "noul", "instructions": QUESTION.format(candidate=c)}
            for i, c in enumerate(candidates)
        },
    }
    request = urllib.request.Request(
        JEV_URL,
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
