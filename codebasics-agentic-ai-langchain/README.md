# Codebasics — Agentic AI & LangChain

> Hands-on practice following the [Codebasics Agentic AI / LangChain video course](https://www.youtube.com/watch?v=D74el9mvNak).

This folder contains my learning exercises and implementations associated with the course. The goal is to understand the components behind AI applications, not just run examples: LLM integration, retrieval, tool use, agent workflows, and practical application patterns.

**Status:** Learning in progress. The sections below describe the course's intended scope; they do not imply every exercise has already been implemented.

## Learning scope

| Area             | What I'm exploring                                                          |
| ---------------- | --------------------------------------------------------------------------- |
| LLM fundamentals | Model calls, prompts, responses, and application integration                |
| LangChain        | Reusable components and orchestration of LLM-powered workflows              |
| RAG              | Document ingestion, chunking, embeddings, retrieval, and grounded responses |
| Vector search    | Embedding-based similarity search and vector storage                        |
| Agents and tools | Tool invocation, reasoning/action loops, and multi-step tasks               |
| Evaluation       | Answer quality, hallucinations, faithfulness, cost, and latency             |

Some topics may be extended beyond the original course as independent practice.

## Guided application scenarios

### Shopping agent

A practical agent scenario exploring how an AI workflow can interpret a shopping request, use supporting tools or data, and return a useful result. Implementation details and exact launch commands should be documented alongside the code as it is added.

### Telecom support RAG chatbot

A support-assistant scenario exploring retrieval of relevant knowledge before generating responses. The emphasis is on grounding answers in retrieved material and understanding the limits of that approach.

These scenarios describe the course practice areas, not verified deployment claims.

## Folder organization

This track lives directly under the `labs-agentic-ai` repository root:

```text
labs-agentic-ai/
└── codebasics-agentic-ai-langchain/
    ├── README.md
    └── ... course notebooks, scripts, and project files as added
```

Code and notebooks can be grouped by topic or application when useful. This README intentionally does not prescribe subdirectories that may not exist.

## Getting started

Navigate to the course folder:

```bash
cd ~/Desktop/my-workspace/github/labs-agentic-ai/codebasics-agentic-ai-langchain
```

### Install `uv`

Check that [`uv`](https://docs.astral.sh/uv/) is installed:

```bash
uv --version
```

If it is missing, follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Recommended: synchronize a `uv` project

If this directory contains a `pyproject.toml`, use **`uv sync`** to create or update the project environment and install the declared dependencies:

```bash
uv sync
source .venv/bin/activate

# Verify the environment
which python
python --version
uv pip list
```

`uv sync` also respects the project's `uv.lock` file when present, helping reproduce dependency versions. **Activation is optional for `uv sync`**: the command manages the project's `.venv` itself. If you prefer to activate the environment first, this works too:

```bash
source .venv/bin/activate
uv sync
```

To add a new dependency to a `uv` project, use `uv add <package-name>` rather than only `uv pip install`, so the project metadata and lockfile stay in sync.

### Alternative: requirements-based exercises

If an exercise has `requirements.txt` but **no `pyproject.toml`**, use a virtual environment with `uv pip` instead:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

You can also install individual packages with `uv pip install <package-name>`. This is a separate workflow from `uv sync` and does not update `pyproject.toml`.

### Running exercises and handling credentials

Run the relevant script or notebook according to the instructions in its own directory. In a `uv` project, `uv run <script.py>` is another option that uses the synchronized project environment without requiring manual activation.

Configure provider API keys through environment variables or a local, ignored `.env` file; **never commit credentials**.

To leave an activated virtual environment:

```bash
deactivate
```

> There is no universal launch command for this course: individual notebooks and applications may have different entry points. The `uv sync` workflow requires a `pyproject.toml` in the project directory; if each example has its own `pyproject.toml`, run it from that example's directory.

## Progress and documentation

For each completed example, I aim to record:

- The problem being solved and why an LLM or agent is appropriate.
- Tools, libraries, and external services used.
- Required environment variables and local setup.
- How to execute and test the example.
- Observed limitations, trade-offs, and potential improvements.

## References and attribution

- **Original course:** [Codebasics Agentic AI / LangChain video](https://www.youtube.com/watch?v=D74el9mvNak)
- **Parent repository:** [labs-agentic-ai](https://github.com/sammsulhoq/labs-agentic-ai)

This is a personal follow-along and experimentation workspace, not an official Codebasics repository. Original teaching materials and examples belong to their respective creators.
