# Agentic AI Labs

Hands-on explorations of **generative AI, retrieval-augmented generation (RAG), AI agents, and agentic workflows** using Python and modern AI development frameworks.

This repository is a practical engineering workspace: guided course exercises live alongside small experiments and independently built projects. Implementations are organized by purpose so each can evolve without turning the repository into a collection of unrelated root-level files.

> **Status:** An evolving learning and experimentation repository. Examples are not presented as production-ready systems unless explicitly stated.

## Repository layout

```text
labs-agentic-ai/
├── courses/                              # Guided implementations from learning resources
│   └── codebasics-agentic-ai-langchain/   # Codebasics Agentic AI / LangChain crash course
│       ├── README.md
│       ├── fundamentals/                  # Core examples and concepts
│       ├── shopping-agent/                # Guided shopping-agent project
│       └── telecom-rag-chatbot/           # Guided telecom support RAG project
├── projects/                             # Independently designed applications
├── experiments/                          # Small, focused proofs of concept
├── .gitignore
└── README.md
```

Folders are added as work begins; not every directory shown above will exist immediately.

## How to navigate

| Area | Purpose | What belongs here |
| --- | --- | --- |
| [`courses/`](courses/) | Follow-along implementations | Exercises, notes, and projects associated with a named course |
| `projects/` | Independent builds | Applications with their own requirements, architecture, and setup |
| `experiments/` | Focused technical exploration | Small comparisons, prototypes, or single-concept demonstrations |

Each substantial project should include its own README describing what it does, the tools it uses, setup steps, and any known limitations.

## Current course

### Codebasics — Agentic AI using LangChain

**Location:** [`courses/codebasics-agentic-ai-langchain/`](courses/codebasics-agentic-ai-langchain/)

**Video:** [Agentic AI crash course on YouTube](https://www.youtube.com/watch?v=D74el9mvNak)

Topics and tools described for the course include LangChain, generative AI foundations, agentic workflows, RAG, ChromaDB, Streamlit, SQLite, Hugging Face, sentence transformers, and LLM integrations. The guided projects are a **shopping agent** and a **telecom RAG chatbot**.

See the [course README](courses/codebasics-agentic-ai-langchain/README.md) for its working layout and progress checklist.

## Running an example

Examples may have different dependencies, API providers, and launch commands. Start with the README in the relevant course or project folder rather than assuming a single setup command works for the entire repository.

For Python examples, a typical local environment is:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install dependencies using the requirements file for the specific example.
```

Never commit API keys, credentials, local databases containing personal data, or `.env` files. Provide `.env.example` files with variable names only when configuration is needed.

## Engineering approach

- Keep examples small, readable, and reproducible.
- Separate tutorial-derived implementations from independently designed projects.
- Document dependencies, assumptions, limitations, and attribution.
- Prefer focused commits that reflect meaningful progress.
- Add tests and deployment instructions when an example warrants them.

## Attribution

Course exercises are inspired by the respective authors and linked to their original learning materials. This repository contains personal practice and adaptations; it is not an official Codebasics repository.
