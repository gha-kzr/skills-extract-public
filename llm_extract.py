"""LLM extraction: ask a chat model for the skills in the offer.

Works with any OpenAI-compatible endpoint (hosted API or local Ollama).
Reads OPENAI_BASE_URL, OPENAI_API_KEY and LLM_MODEL from the environment.

Usage: uv run --env-file .env llm_extract.py --offer offer.txt [--model <name>]
"""
import argparse
import os

from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--offer", default="offer.txt")
parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
args = parser.parse_args()
if not args.model:
    parser.error("pass --model or set LLM_MODEL")

text = open(args.offer, encoding="utf-8").read()

client = OpenAI()  # picks up OPENAI_BASE_URL and OPENAI_API_KEY

response = client.chat.completions.create(
    model=args.model,
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract the technical skills (tools, languages, platforms, "
                "methods) named in this job offer. Use canonical English "
                "names (e.g. 'Kubernetes'). Answer with one skill per line, "
                "nothing else."
            ),
        },
        {"role": "user", "content": text},
    ],
)
print(response.choices[0].message.content)
