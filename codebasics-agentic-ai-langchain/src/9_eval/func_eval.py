"""Evaluate the inventory agent using a LangSmith dataset."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client, evaluate, traceable
from langsmith.schemas import Example, Run

from inventory_agent import MODEL_NAME, run_inventory_agent
from utils import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent

DATASET_NAME = "inventorydata"
EXPERIMENT_PREFIX = (
    f"inventory_agent_evaluation_{MODEL_NAME.rsplit('/', 1)[-1]}"
)

# Each example contains an input question and its reference answer.
EVALUATION_EXAMPLES = [
    {
        "inputs": {
            "question": "What is the stock status of iPhone 15?",
        },
        "outputs": {
            "answer": (
                "The iPhone 15 is currently in stock "
                "with 2 units available."
            ),
        },
    },
    {
        "inputs": {
            "question": "Is AirPods Pro available?",
        },
        "outputs": {
            "answer": (
                "The AirPods Pro is currently out of stock. "
                "There are 0 available items."
            ),
        },
    },
    {
        "inputs": {
            "question": "How many iPhone 15 units are available?",
        },
        "outputs": {
            "answer": (
                "The iPhone 15 is currently in stock "
                "with 2 units available."
            ),
        },
    },
    {
        "inputs": {
            "question": "Do you have Samsung Galaxy S23?",
        },
        "outputs": {
            "answer": "The product is not available in our inventory.",
        },
    },
    {
        "inputs": {
            "question": "Can you tell me the recipe of Vada Pav?",
        },
        "outputs": {
            "answer": "Sorry, I can’t assist with that.",
        },
    },
]


@traceable
def inventory_evaluation_target(inputs: dict) -> dict:
    """Run the application for one dataset input."""
    question = inputs["question"]
    generated_answer = run_inventory_agent(question)

    return {"answer": generated_answer}


def ensure_evaluation_dataset(langsmith_client: Client) -> None:
    """Create and populate the dataset if it does not already exist."""
    if langsmith_client.has_dataset(dataset_name=DATASET_NAME):
        # Reuse existing examples without adding duplicates.
        return

    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME,
    )

    langsmith_client.create_examples(
        dataset_id=dataset.id,
        examples=EVALUATION_EXAMPLES,
    )


def semantic_match(run: Run, example: Example) -> dict:
    """Compare a generated answer with its reference using embeddings."""
    # LangSmith uses the parameter names "run" and "example"
    # when supplying evaluator arguments.
    if example.outputs is None or run.outputs is None:
        raise ValueError(
            "Evaluation requires reference outputs and generated outputs."
        )

    reference_answer = example.outputs["answer"]
    generated_answer = run.outputs["answer"]

    similarity_score = cosine_similarity(
        reference_text=reference_answer,
        generated_text=generated_answer,
    )

    # Preserve the original evaluator's nonnegative scoring behavior.
    # The utility itself returns the full cosine range of [-1, 1].
    evaluation_score = max(0.0, similarity_score)

    return {
        "key": "semantic_match",
        "score": evaluation_score,
    }


def main() -> None:
    """Prepare the dataset and launch a LangSmith evaluation experiment."""
    load_dotenv(BASE_DIR / ".env")

    for variable_name in ("GROQ_API_KEY", "LANGSMITH_API_KEY"):
        if not os.getenv(variable_name):
            raise ValueError(
                f"Set {variable_name} in your environment or .env file."
            )

    langsmith_client = Client()

    ensure_evaluation_dataset(langsmith_client)

    evaluate(
        inventory_evaluation_target,
        client=langsmith_client,
        data=DATASET_NAME,
        evaluators=[semantic_match],
        experiment_prefix=EXPERIMENT_PREFIX,
        # Keep this small for the demo and to reduce concurrent API requests.
        max_concurrency=1,
    )


# Importing this module does not create datasets or launch evaluations.
if __name__ == "__main__":
    main()
