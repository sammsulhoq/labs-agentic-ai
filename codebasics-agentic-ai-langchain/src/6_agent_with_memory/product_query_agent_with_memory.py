"""Demonstrate product agents with tools and conversation memory."""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver


# Shared model and conversation settings.
MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE = 0
CONVERSATION_THREAD_ID = "user-alice-session-1"

SYSTEM_PROMPT = """
You are a helpful product assistant for an online tech store.
Use the available tools to look up product information.
Do not invent prices, ratings, reviews, currency, or stock availability.
Distinguish catalog ratings from customer review ratings when they differ.
If a requested detail is unavailable, say so clearly.
Use conversation history to understand follow-up questions.
"""

# Sample catalog data. No currency or stock information is provided.
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

# Review summaries are separate from catalog information.
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
    normalized_name = product_name.strip().lower()
    product_details = PRODUCT_CATALOG.get(normalized_name)

    if product_details is None:
        available_products = ", ".join(PRODUCT_CATALOG)
        return (
            f"Product not found. Available products: {available_products}"
        )

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


def ask_agent(
    agent: Runnable,
    question: str,
    thread_id: str | None = None,
) -> None:
    """Send a question, optionally using a saved conversation thread."""
    config: RunnableConfig = {}

    if thread_id is not None:
        # The checkpointer uses this ID to locate conversation state.
        config["configurable"] = {"thread_id": thread_id}

    # Send only the new question. The checkpointer restores prior messages
    # automatically when the same thread ID is used.
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=config,
    )

    final_response = result["messages"][-1]

    print(f"\nQuestion: {question}")
    print(f"Answer: {final_response.content}")


def main() -> None:
    """Run a single-tool example and a multi-tool conversation with memory."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "Set GROQ_API_KEY in your environment or .env file."
        )

    # Both agents reuse the same model instance.
    # Conversation memory belongs to the checkpointer, not the model.
    chat_model = ChatGroq(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
    )

    # Example 1: A single-tool agent without conversation memory.
    product_agent = create_agent(
        model=chat_model,
        tools=[get_product_details],
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n=== EXAMPLE 1: PRODUCT DETAILS WITHOUT MEMORY ===")

    product_questions = [
        "What is the price of wireless headphones?",
        "Tell me about the mechanical keyboard.",
    ]

    for question in product_questions:
        ask_agent(product_agent, question)

    # Create the checkpointer once and reuse it throughout the conversation.
    # Creating a new InMemorySaver for every question would lose history.
    conversation_memory = InMemorySaver()

    # Example 2: A two-tool agent with conversation memory.
    product_and_reviews_agent = create_agent(
        model=chat_model,
        tools=[get_product_details, get_product_reviews],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=conversation_memory,
    )

    print("\n=== EXAMPLE 2: PRODUCT DETAILS AND REVIEWS WITH MEMORY ===")

    review_questions = [
        "How do people like the smart watch?",
        "What are the price and customer reviews of the smart watch?",
        "How many people reviewed it?",
        "What is its battery life?",
    ]

    for question in review_questions:
        # Reuse this ID so "it" and "its" refer to the previous product.
        ask_agent(
            agent=product_and_reviews_agent,
            question=question,
            thread_id=CONVERSATION_THREAD_ID,
        )


# Importing this module does not create agents or make API calls.
if __name__ == "__main__":
    main()
