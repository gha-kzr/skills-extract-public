# skills-extract (minimal demo)

Companion code of the talk *Un dictionnaire peut-il encore battre ton IA ?*
(BDX I/O 2026). Four ways to extract the skills of a job offer, each in a
short script (its `main`) that prints the skills to the console. The helper
functions are in [`src/lib.py`](src/lib.py):

| Script | Approach |
|---|---|
| [`src/dict_extract.py`](src/dict_extract.py) | 1. Dictionary lookup (a list of known skills) |
| [`src/ner_extract.py`](src/ner_extract.py) | 2. Pretrained NER model (Hugging Face) |
| [`src/llm_extract.py`](src/llm_extract.py) | 3. Chat LLM through an OpenAI-compatible API |
| [`src/jev_extract.py`](src/jev_extract.py) | 4. Decision model ("System One", Jev): one yes/no question per n-gram |

The sample offer is `data/offer.txt`; the expected output is a list like
`Kubernetes, Terraform, ArgoCD, Docker, GitHub Actions, Prometheus, Grafana, AWS, GCP, Linux...`

## Quick start

```bash
git clone <this repo> && cd skills-extract/public

# 1. install uv (once): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. run the approaches that need no configuration
uv run src/dict_extract.py                        # works offline
uv run src/ner_extract.py                         # first run downloads a ~1 GB model

# 3. for the LLM and Jev approaches, create your env file and fill it
cp .env.example .env                              # see the table below
uv run --env-file .env src/llm_extract.py         # needs OPENAI_* and LLM_MODEL
uv run --env-file .env src/jev_extract.py         # needs TYPESAFE_API_KEY
```

`uv run` creates the virtual environment and installs the dependencies by
itself. Run the commands from the root of the repo. Every script takes
`--offer <file>` to use another offer (default: `data/offer.txt`); `--help`
lists the other options.

### Configuration (`.env`)

| Variable | Used by | Value |
|---|---|---|
| `OPENAI_BASE_URL` | `src/llm_extract.py` | endpoint of an OpenAI-compatible API |
| `OPENAI_API_KEY` | `src/llm_extract.py` | its token (any non-empty value for Ollama) |
| `LLM_MODEL` | `src/llm_extract.py` | model name (or pass `--model`) |
| `TYPESAFE_API_KEY` | `src/jev_extract.py` | API key from [typesafe.ai](https://typesafe.ai) |

For a **free local LLM** with [Ollama](https://ollama.com):

```bash
ollama pull gemma3:4b
# .env: OPENAI_BASE_URL=http://localhost:11434/v1  OPENAI_API_KEY=ollama  LLM_MODEL=gemma3:4b
```

For a **hosted LLM**, set `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `LLM_MODEL` to
your provider's values. `.env` is git-ignored: never commit it.

## The four approaches

Specific options: `--dictionary data/skills.txt` (one skill per line) for the
dictionary, `--threshold 0.8` for Jev.

### 1. Dictionary

Code: [`src/dict_extract.py`](src/dict_extract.py)

A list of known skills (python, kubernetes, dbt...) compared word by word with
the text of the offer.

```python
for word in offer_text.split():
    if word in dictionary:
        skills.add(word)
```

The demo dictionary is `data/skills.txt`, one skill per line, matched as whole
words, in any case. Everything depends on the quality of the dictionary. Limits: no context ("C",
"less" or "vite" are also ordinary words), the dictionary must be enriched or
rebuilt regularly, synonyms and compound names (k8s / kubernetes) must be
listed, similar skills are not grouped.

### 2. ML / NER

Code: [`src/ner_extract.py`](src/ner_extract.py), helpers `find_spans` and `whole_words` in [`src/lib.py`](src/lib.py)

A multilingual model pretrained on the ESCO skill taxonomy
([`jjzha/escoxlmr_knowledge_extraction`](https://huggingface.co/jjzha/escoxlmr_knowledge_extraction))
tags each word of the offer as Beginning, Inside or Outside a skill:

```
Vous  maîtrisez  Apache  Kafka  et  Python
 O        O        B       I     O     B
```

```python
for word, tag in NER_model(offer_text):
    if tag in ("B", "I"):
        skills.add(word)
```

The script groups consecutive `B`/`I` tokens into one skill ("Apache Kafka"),
drops the ones the model is not confident about (average score below 0.7) and
widens them to whole words. No retraining needed (it detects tools newer than
the model). Limits: the
boundary between a skill and a domain (conception, finance) is fuzzy, and the
model is hard to extend to other problems.

### 3. LLM

Code: [`src/llm_extract.py`](src/llm_extract.py)

One call to a chat model (hosted or local) with a prompt.

```python
PROMPT = """Extract the technical skills of this job offer.
Return a structured JSON: [{"skills": ["skill1", "skill2"]}]"""

response = llm.chat(
    system=PROMPT,
    user=offer_text,
)
skills = json.loads(response)  # [{"skills": ["Kubernetes", ...]}]
```

(The demo script asks for one skill per line instead of JSON, to stay short.)
What matters: the model (and thinking mode), the prompt, the response schema,
the temperature. Limits: cost (to put in perspective), slowest approach when run
locally, variable reliability.

### 4. Decision model (System One / Jev)

Code: [`src/jev_extract.py`](src/jev_extract.py), helpers `generate_candidates` and `ask_jev` in [`src/lib.py`](src/lib.py)

One call to a *decision model*, similar to an LLM but it answers questions with
a probability instead of generating text. The offer is sent once as the state,
with one question per candidate.

```python
offer_text = "une expérience sur kubernetes"

answers = decision_model(
    state=offer_text,
    questions=[
        "Est-ce que « une » est une compétence technique ?",
        "Est-ce que « expérience » est une compétence technique ?",
        "Est-ce que « sur » est une compétence technique ?",
        "Est-ce que « kubernetes » est une compétence technique ?",
    ],
)
# answers = [0.01, 0.05, 0.01, 0.98]

skills = answers > 0.8    # [False, False, False, True]
```

In the script, the candidates are **all the 1 and 2 word n-grams** of the
offer, minus those that start or end with a function word ("avec Terraform"), and the
model is [Jev](https://jevai.wiki/) by TypeSafe AI
([announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)).
Overlapping answers are expected ("GitHub", "Actions" and "GitHub Actions" can
all pass the threshold): the script does not merge them.

The real question becomes: how to propose the best candidates without sending
every possible n-gram? Answer: a hybrid approach (dictionary and/or NER to
propose candidates, decision model to judge them). That is not in this minimal
demo.

## Results from the talk

1000 offers, scored against a manual reference:

| Method | Recall | Precision | F1 | Time | Cost |
|---|---|---|---|---|---|
| Dictionary (scraped) | 0.71 | 0.42 | 0.53 | ~2 s | free |
| ML / NER | 0.74 | 0.67 | 0.70 | ~3 min | free |
| Local LLM (gemma3:4b) | 0.50 | 0.80 | 0.62 | ~1 h 26 | free |
| LLM API (deepseek-v4.1-flash) | 0.89 | 0.78 | 0.83 | ~3 min | $0.42 |
| Decision model (jev-1.14) | 0.79 | 0.87 | 0.83 | ~26 s | $0.37 |

The demo scripts are simplified versions: expect lower scores than these
tuned pipelines.

## What's next

Approaches not covered here:

1. **Embeddings**: match offer passages to skills by vector similarity.
2. **OpenSearch**: index the skill dictionary and offers and query them with search.
3. **Hybrid**: a cheap method proposes candidates, the decision model judges them.
