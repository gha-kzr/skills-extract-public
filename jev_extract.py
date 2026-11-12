"""Decision-model extraction: ask Jev (a System One model) about every 1-2 word n-gram.

The offer is sent once as the `state`; each n-gram is one yes/no question
("is X a technology, tool, technical concept or method?") answered with a
probability. We keep the n-grams above a threshold.

Reads TYPESAFE_API_KEY from the environment.

Usage: uv run --env-file .env jev_extract.py --offer offer.txt
"""
import argparse
import json
import os
import re
import urllib.request

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
QUESTION = 'Is "{c}" a technology, tool, technical concept or method?'

parser = argparse.ArgumentParser()
parser.add_argument("--offer", default="offer.txt")
parser.add_argument("--threshold", type=float, default=0.8)
args = parser.parse_args()

text = open(args.offer, encoding="utf-8").read()

# 1. Candidates: every 1-word and 2-word n-gram of the offer, except those
# starting or ending with a function word ("avec Terraform", "ArgoCD sur").
FUNCTION_WORDS = set(
    "le la les l un une des de d du au aux et ou en dans sur sous par pour avec sans "
    "via notre nos votre vos ce cette ces que qui à a the an of and or in on for with "
    "to by from at as is are our your you we".split()
)
# Separators become spaces ("Prometheus/Grafana" -> two words); c++, c#,
# node.js and ci/cd stay single tokens.
cleaned = re.sub(r"(ci/cd|c\+\+|c#)|[/:;,()\[\]|]", lambda m: m.group(1) or " ", text, flags=re.I)
tokens = re.findall(r"[\w+#]+(?:[./][\w+#]+)*", cleaned)
candidates = {}  # case-insensitive dedupe, keeps the first spelling
for n in (1, 2):
    for i in range(len(tokens) - n + 1):
        words = tokens[i : i + n]
        if words[0].lower() in FUNCTION_WORDS or words[-1].lower() in FUNCTION_WORDS:
            continue
        gram = " ".join(words)
        candidates.setdefault(gram.lower(), gram)
candidates = list(candidates.values())

# 2. One request: the offer once, one question per candidate.
payload = {
    "model": MODEL,
    "state": f"{DEFINITION}\n\nJOB OFFER:\n{text}",
    "questions": {
        f"c{i}": {"type": "noul", "instructions": QUESTION.format(c=c)}
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

# 3. Keep the candidates whose probability of "yes" is above the threshold.
for i, candidate in enumerate(candidates):
    if answers[f"c{i}"]["noul"] >= args.threshold:
        print(candidate)
