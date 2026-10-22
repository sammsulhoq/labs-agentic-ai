# Agentic AI Labs

> A growing collection of hands-on implementations, course exercises, and technical experiments in generative AI and agentic systems.

This repository is my practical workspace for exploring **LLM applications, retrieval-augmented generation (RAG), AI agents, and agentic workflows** with Python and related tools. It brings together learning resources and hands-on implementations without presenting tutorial exercises as production-ready products.

## Repository structure

Each substantial course or learning track lives in a **descriptively named folder directly under the repository root**. Individual folders own their code, dependencies, setup instructions, and README. There is no mandatory `courses/` wrapper.

```text
labs-agentic-ai/
├── codebasics-agentic-ai-langchain/
│   └── README.md
├── <future-course-or-project>/
│   └── README.md
└── README.md
```

*The second directory is illustrative, not an existing folder. Subfolders within each project depend on the implementation.*

## Learning tracks

| Track | Focus | Resource |
| --- | --- | --- |
| [Codebasics — Agentic AI & LangChain](./codebasics-agentic-ai-langchain/) | LangChain, RAG, tools, agents, and practical AI applications | [YouTube course](https://www.youtube.com/watch?v=D74el9mvNak) |

More tracks will be added as hands-on work progresses.

## How to use this repository

1. Open a track from the table above.
2. Read its own `README.md` for requirements, setup, and run commands.
3. Work inside that track's folder; examples may use different environments or dependencies.

For a Python-based example, a common starting point is:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies only from the relevant project's documented requirements. There is no repository-wide installation command.

## Organization principles

- **Root-level folders identify a course, project, or coherent learning track.**
- **Each track is self-contained:** code, supporting assets, and documentation stay together.
- **Keep attribution clear:** link to original courses and distinguish guided exercises from independent extensions.
- **Document actual behavior:** include run commands and environment variables when implementations are available.
- **Protect secrets:** never commit API keys, `.env` files, tokens, credentials, or private datasets.
- **Keep experiments honest:** examples are for learning unless explicitly documented and tested for production use.

## Topics explored

Python · Generative AI · LangChain · LangGraph · RAG · Vector Databases · LLM Integrations · Tool Calling · Agentic Workflows · Evaluation

These are areas of interest across the repository, not a claim that every topic has already been implemented.

## Attribution

Third-party courses and learning resources remain the work of their respective creators. Exercises here are personal implementations or adaptations, not official course repositories.

## Author

[Sammsul Hoque Choudhary](https://github.com/sammsulhoq)
