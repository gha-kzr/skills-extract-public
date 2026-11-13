"""LLM extraction: ask a chat model for the skills in the offer.

Works with any OpenAI-compatible endpoint (hosted API or local Ollama).
Reads OPENAI_BASE_URL, OPENAI_API_KEY and LLM_MODEL from the environment.

Usage: uv run --env-file .env src/llm_extract.py [--offer data/offer.txt] [--model <name>]
"""
from pathlib import Path

import typer
from openai import OpenAI

DATA = Path(__file__).resolve().parent.parent / "data"

PROMPT = (
    "Extract the technical skills (tools, languages, platforms, methods) named "
    "in this job offer. Use canonical English names (e.g. 'Kubernetes'). "
    "Answer with one skill per line, nothing else."
)


def extract(text: str, model: str) -> str:
    client = OpenAI()  # picks up OPENAI_BASE_URL and OPENAI_API_KEY
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


def main(
    offer: Path = DATA / "offer.txt",
    model: str = typer.Option(..., envvar="LLM_MODEL", help="Model name"),
):
    print(extract(offer.read_text(encoding="utf-8"), model))


if __name__ == "__main__":
    typer.run(main)
