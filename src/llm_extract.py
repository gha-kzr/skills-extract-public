"""LLM extraction: ask a chat model for the skills in the offer.

Works with any OpenAI-compatible endpoint (hosted API or local Ollama).
Reads OPENAI_BASE_URL, OPENAI_API_KEY and LLM_MODEL from the environment.

Usage: uv run --env-file .env src/llm_extract.py [--offer data/offer.txt] [--model <name>]
"""
from pathlib import Path

import typer
from openai import OpenAI

from lib import DATA

PROMPT = (
    "Extract the technical skills (tools, languages, platforms, methods) named "
    "in this job offer. Use canonical English names (e.g. 'Kubernetes'). "
    "Answer with one skill per line, nothing else."
)


def main(
    offer: Path = DATA / "offer.txt",
    model: str = typer.Option(..., envvar="LLM_MODEL", help="Model name"),
):
    text = offer.read_text(encoding="utf-8")

    client = OpenAI()  # picks up OPENAI_BASE_URL and OPENAI_API_KEY
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    typer.run(main)
