# LLM Foundations: Prompting, RAG, Tool Calling, Embeddings, and Evaluation

A hands-on project with five Jupyter notebooks and the supporting Python code. It covers
the building blocks of LLM applications and how to test them.

The notebooks run **offline by default**, with no API key and no downloads. A deterministic
mock model and a lexical embedder stand in for real services. Switch to a real provider
with environment variables when you want to measure real quality.

## Contents

```
llm_project/
├── notebooks/
│   ├── 01_prompt_engineering.ipynb
│   ├── 02_embeddings_and_vector_stores.ipynb
│   ├── 03_retrieval_augmented_generation.ipynb
│   ├── 04_function_tool_calling.ipynb
│   └── 05_evaluation_frameworks.ipynb
├── src/
│   ├── llm_utils.py      # chat clients: MockLLM (offline) and OpenAIChat
│   ├── embeddings.py     # hashed (offline) and OpenAI embeddings, cosine similarity
│   ├── vector_store.py   # in-memory vector store with filters and save/load
│   ├── rag.py            # chunking and the RAG pipeline with citations
│   ├── tools.py          # tool registry, validation, safe arithmetic, tool loop
│   ├── prompting.py      # checked prompt templates, few-shot builder, JSON parsing
│   ├── evaluation.py     # metrics, retrieval scores, groundedness, judge parsing, runner
│   └── sample_data.py    # shared sample corpus and evaluation cases
├── tests/
│   └── test_core.py      # unit tests for the src modules
├── build_notebooks.py    # generates the notebooks from Python source
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9 or newer.

```bash
cd llm_project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core code uses only the standard library. `jupyter` and `pytest` are for running
the notebooks and tests. `openai` is optional.

## Running the notebooks

```bash
jupyter notebook notebooks/
```

Run the notebooks in order. Each one adds `src/` to the import path itself, so run them
from the project folder or from `notebooks/`.

| # | Notebook | You will learn |
|---|---|---|
| 01 | Prompt engineering | Checked templates, few-shot messages, validating JSON output, comparing prompt versions |
| 02 | Embeddings and vector stores | Cosine similarity, indexing, top-k search, metadata filters, persistence |
| 03 | Retrieval-augmented generation | Chunking, retrieval metrics, answers with citations, abstaining when the context is missing |
| 04 | Function (tool) calling | Schemas, argument validation, safe execution, the tool loop, production safeguards |
| 05 | Evaluation frameworks | Metrics, a runner, release-gate thresholds, judge plumbing, regression practice |

To regenerate the notebooks after editing the build script:

```bash
python3 build_notebooks.py
```

## Running the tests

```bash
pytest -q
```

Or without pytest:

```bash
python3 tests/test_core.py
```

## Switching to a real model or embeddings

Set environment variables before starting Jupyter or running a script:

```bash
export OPENAI_API_KEY="your-key"       # never commit this
export LLM_PROVIDER=openai             # default: mock
export LLM_MODEL=gpt-4o-mini           # optional
export EMBED_PROVIDER=openai           # default: hash
export EMBED_MODEL=text-embedding-3-small   # optional
```

Notebooks 01 to 05 use the same code paths. Expect real differences in behaviour:

- Prompt wording changes results, so the comparison in notebook 01 becomes meaningful.
- Semantic embeddings match paraphrases, so notebook 02's similarity table changes.
- Real models may answer out-of-scope questions from general knowledge, so the abstention checks in notebooks 03 and 05 matter more.
- Judge scoring (notebook 05, section 4) works only with a real provider.

To add another provider, implement a `chat(messages, tools=None)` method that returns a
`ChatResponse` with `content` and `tool_calls`, and register it in `get_llm()`.

## Key design points

- **Mock models are for plumbing, not quality.** They test prompts, retrieval, tool loops, and parsing. Their answers do not reflect real model behaviour, and the notebooks say so where it matters.
- **Retrieval and generation are evaluated separately.** A wrong answer can come from either step, so each has its own metrics.
- **Tools are validated before they run.** The model proposes calls. Your code checks the arguments, runs the function, and reports structured errors back.
- **Arithmetic is evaluated without `eval`.** The calculator only accepts numbers and four operators.
- **Evaluation is a gate, not a report.** Thresholds decide whether a change ships.

## Testing the notebooks

Each notebook's code cells run top to bottom without errors on the offline defaults. To
check this yourself:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

This requires `jupyter` (installed above).

## Known limitations

- The in-memory vector store does a full scan. It suits small corpora only. Use FAISS, Chroma, pgvector, or a managed service for large data.
- The hashed embedder is lexical. It cannot match paraphrases. It exists so the project runs offline.
- The groundedness score is a word-overlap heuristic. It misses subtle errors, so use a validated judge for real decisions.
- The sample corpus and evaluation cases are small and illustrative. Build your own from real user questions before relying on any score.
- Refusal and abstention checks rely on simple patterns and phrases. Test them against your own wording.

## Further reading

- Evaluation and monitoring concepts for GenAI systems: see the companion guide `genai_eval_guide.md` (if included with this project).
- Frameworks for evaluating LLM applications: Ragas, DeepEval, promptfoo, TruLens.
