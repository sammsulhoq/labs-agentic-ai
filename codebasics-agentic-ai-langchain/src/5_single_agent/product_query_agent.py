"""Demonstrate product assistants with one tool and multiple tools."""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq


# Shared configuration for both agents.
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE = 0

SYSTEM_PROMPT = """
You are a helpful product assistant for an online tech store.
Use the available tools to look up product information.
Do not invent prices, ratings, reviews, or stock availability.
Distinguish catalog ratings from customer review ratings when they differ.
If a requested detail is unavailable, say so clearly.
"""

# Sample catalog data. Prices retain the original unspecified currency.
PRODUCT_CATALOG = {
    "wireless headphones": {
        "price": 79.99,
        "rating": 4.6,
        "description": (
            "Over-ear Bluetooth, 30-hour battery, "
            "active noise cancellation."
        ),
    },
    "smart watch": {
        "price": 199.99,
        "rating": 4.3,
        "description": (
            "Tracks heart rate and sleep. "
            "5-day battery, water-resistant."
        ),
    },
    "mechanical keyboard": {
        "price": 129.00,
        "rating": 4.8,
        "description": (
            "Tenkeyless, Cherry MX Brown switches, per-key RGB."
        ),
    },
    "laptop stand": {
        "price": 34.99,
        "rating": 4.5,
        "description": (
            "Adjustable aluminium, fits 11–17 inch laptops, "
            "folds flat."
        ),
    },
}

# Review summaries are separate from the catalog.
# The smart watch ratings intentionally preserve your original values.
PRODUCT_REVIEWS = {
    "wireless headphones": {
        "review_count": 1262,
        "rating": 4.6,
    },
    "smart watch": {
        "review_count": 340,
        "rating": 3.9,
    },
    "mechanical keyboard": {
        "review_count": 67,
        "rating": 4.8,
    },
    "laptop stand": {
        "review_count": 781,
        "rating": 4.5,
    },
}


@tool
def get_product_details(product_name: str) -> str:
    """Return a product's price, catalog rating, and description by name."""
    # Ignore capitalization and extra surrounding whitespace.
    normalized_name = product_name.strip().lower()
    product_details = PRODUCT_CATALOG.get(normalized_name)

    if product_details is None:
        available_products = ", ".join(PRODUCT_CATALOG)
        return (
            f"Product not found. Available products: {available_products}"
        )

    # JSON gives the model clearly named fields rather than a Python repr.
    return json.dumps(
        {
            "product_name": normalized_name,
            "price": product_details["price"],
            "catalog_rating": product_details["rating"],
            "description": product_details["description"],
        }
    )


@tool
def get_product_reviews(product_name: str) -> str:
    """Return a product's name, review count, and customer review rating."""
    normalized_name = product_name.strip().lower()
    review_summary = PRODUCT_REVIEWS.get(normalized_name)

    if review_summary is None:
        return f"No reviews available for '{product_name}'."

    return json.dumps(
        {
            "product_name": normalized_name,
            "review_count": review_summary["review_count"],
            "customer_review_rating": review_summary["rating"],
        }
    )


def ask_agent(agent: Runnable, question: str) -> None:
    """Send a question to an agent and display its final response."""
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    # The result includes the messages generated during tool execution.
    # The final message contains the assistant's answer.
    final_response = result["messages"][-1]

    print(f"\nQuestion: {question}")
    print(f"Answer: {final_response.content}")


def main() -> None:
    """Initialize the model and demonstrate both tool configurations."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "Set GROQ_API_KEY in your environment or .env file."
        )

    # Reuse one model instance across both agents.
    chat_model = ChatGroq(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
    )

    # Example 1: The agent can look up catalog details.
    product_agent = create_agent(
        model=chat_model,
        tools=[get_product_details],
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n=== EXAMPLE 1: PRODUCT DETAILS TOOL ===")

    product_questions = [
        "What is the price of wireless headphones?",
        "Tell me about the mechanical keyboard.",
    ]

    for question in product_questions:
        ask_agent(product_agent, question)

    # Example 2: The agent can choose between two tools or use both.
    product_and_reviews_agent = create_agent(
        model=chat_model,
        tools=[get_product_details, get_product_reviews],
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n=== EXAMPLE 2: PRODUCT DETAILS AND REVIEWS TOOLS ===")

    review_questions = [
        "How do people like the smart watch?",
        "What are the price and customer reviews of the smart watch?",
    ]

    for question in review_questions:
        ask_agent(product_and_reviews_agent, question)


# Importing this module does not initialize agents or make API calls.
if __name__ == "__main__":
    main()
