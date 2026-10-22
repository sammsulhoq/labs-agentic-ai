# Codebasics — Agentic AI & LangChain

> Hands-on practice following the [Codebasics Agentic AI / LangChain video course](https://www.youtube.com/watch?v=D74el9mvNak).

This folder contains my learning exercises and implementations associated with the course. The goal is to understand the components behind AI applications, not just run examples: LLM integration, retrieval, tool use, agent workflows, and practical application patterns.

**Status:** Learning in progress. The sections below describe the course's intended scope; they do not imply every exercise has already been implemented.

## Learning scope

| Area | What I'm exploring |
| --- | --- |
| LLM fundamentals | Model calls, prompts, responses, and application integration |
| LangChain | Reusable components and orchestration of LLM-powered workflows |
| RAG | Document ingestion, chunking, embeddings, retrieval, and grounded responses |
| Vector search | Embedding-based similarity search and vector storage |
| Agents and tools | Tool invocation, reasoning/action loops, and multi-step tasks |
| Evaluation | Answer quality, hallucinations, faithfulness, cost, and latency |

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

From the local GitHub workspace, create and activate a Python virtual environment:

```bash
cd ~/Desktop/my-workspace/github/labs-agentic-ai/codebasics-agentic-ai-langchain
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies using `uv`

With `.venv` activated, check whether [`uv`](https://docs.astral.sh/uv/) is installed:

```bash
uv --version
```

If it is not installed, follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) and reopen the terminal if needed. Then reactivate `.venv`.

Use `uv pip` to install dependencies into the **active virtual environment**:

```bash
# If the current exercise has a requirements.txt file
uv pip install -r requirements.txt

# Or install individual packages when needed (example)
uv pip install langchain langchain-community python-dotenv
```

Run only the command relevant to the exercise; the package list above is illustrative, not a verified list of course dependencies. If a requirements file lives inside a specific project subfolder, change into that folder or pass its relative path instead.

To verify which Python environment is active:

```bash
which python
python --version
uv pip list
```

`which python` should point to the `.venv/bin/python` inside this course folder. With an activated environment, `uv pip install` targets that environment; it is a faster alternative to `pip install`.

> **Important:** `uv pip install` is the appropriate command here for an already activated virtual environment. `uv run` is a separate workflow that can manage or select a project environment, so it is not necessary for these steps.

Configure provider API keys through environment variables or a local, ignored `.env` file; **never commit credentials**.

To leave the virtual environment:

```bash
deactivate
```

> No universal application run command is listed because notebooks and applications may have different entry points. Follow the instructions documented with each implementation.

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
