"""Answer inventory questions using a tool-enabled Groq agent."""

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq


BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "qwen/qwen3-32b"
MODEL_TEMPERATURE = 0

logger = logging.getLogger(__name__)

# Lowercase keys support case-insensitive product lookups.
INVENTORY_RECORDS = {
    "iphone 15": {
        "product_name": "iPhone 15",
        "available_units": 2,
    },
    "airpods pro": {
        "product_name": "AirPods Pro",
        "available_units": 0,
    },
    "macbook air m3": {
        "product_name": "MacBook Air M3",
        "available_units": 5,
    },
}

SYSTEM_PROMPT = """
You are an inventory assistant.

For questions unrelated to product inventory, respond exactly:
"Sorry, I can’t assist with that."
Do not call a tool for these questions.

For inventory questions:
- Always call check_inventory before answering.
- Pass the full product name to the tool.
- The tool returns plain text containing the stock status and unit count.
- State the product's stock status and number of available units.
- If the product is not found, respond exactly:
  "The product is not available in our inventory."
- Never guess availability or invent inventory information.

Keep responses short and informative.
"""


@tool
def check_inventory(product_name: str) -> str:
    """Return the stock status and available unit count for a product."""
    normalized_name = product_name.strip().casefold()

    logger.info("Checking inventory for: %s", normalized_name)

    product_record = INVENTORY_RECORDS.get(normalized_name)

    if product_record is None:
        return "Product not found in inventory."

    available_units = product_record["available_units"]
    stock_status = "In Stock" if available_units > 0 else "Out of Stock"

    return (
        f"Product: {product_record['product_name']} | "
        f"Stock status: {stock_status} | "
        f"Available units: {available_units}"
    )


@lru_cache(maxsize=1)
def get_inventory_agent() -> Runnable:
    """Create the agent on first use and reuse it for later questions."""
    load_dotenv(BASE_DIR / ".env")

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "Set GROQ_API_KEY in your environment or .env file."
        )

    chat_model = ChatGroq(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
    )

    return create_agent(
        model=chat_model,
        tools=[check_inventory],
        system_prompt=SYSTEM_PROMPT,
    )


def run_inventory_agent(question: str) -> str:
    """Submit an inventory question and return the final answer."""
    if not question.strip():
        raise ValueError("The question must not be empty.")

    inventory_agent = get_inventory_agent()

    agent_result = inventory_agent.invoke(
        {
            "messages": [
                HumanMessage(content=question),
            ]
        }
    )

    return agent_result["messages"][-1].text


def main() -> None:
    """Run a sample inventory lookup."""
    question = "What is the inventory status of iPhone 15?"
    answer = run_inventory_agent(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
