"""Demonstrate customer lookup with PII masking and redaction."""

import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq


# Shared model configuration.
MODEL_NAME = "qwen/qwen3-32b"
MODEL_TEMPERATURE = 0

# Synthetic customer records for demonstrating PII middleware.
# Credit card values are test numbers, not real payment credentials.
CUSTOMER_RECORDS = {
    "krishna": {
        "name": "Krishna",
        "email": "krishna_001@abc.com",
        "credit_card": "4111-1111-1111-1111",
        "loyalty_status": "Gold",
    },
    "alice": {
        "name": "Alice",
        "email": "alice_002@abc.com",
        "credit_card": "5500-0000-0000-0004",
        "loyalty_status": "Silver",
    },
    "bob": {
        "name": "Bob",
        "email": "bob2009@xyz.com",
        "credit_card": "6011-1111-1111-1117",
        "loyalty_status": "Bronze",
    },
}

SYSTEM_PROMPT = """
You are a customer service assistant.

When asked about a customer, use get_customer_info to retrieve their details.

Response rules:
1. Present all returned customer fields as labeled plain text, not JSON.
2. Preserve field values exactly as supplied in the tool result.
3. Preserve masked values and redaction placeholders without modification.
4. Do not infer, reconstruct, or invent missing or redacted information.
5. If the customer is not found, say so clearly.
"""


@tool
def get_customer_info(customer_name: str) -> str:
    """Look up a customer by name and return their customer details.

    Uses synthetic records to simulate a database or external API lookup.
    """
    # Support names with different capitalization or surrounding whitespace.
    normalized_name = customer_name.strip().casefold()
    customer_record = CUSTOMER_RECORDS.get(normalized_name)

    if customer_record is None:
        return "Customer not found."

    # Serialize the record so the declared return type is consistently str.
    # PII middleware processes the tool-result message before model use.
    return json.dumps(customer_record)


def create_pii_middleware() -> list[PIIMiddleware]:
    """Configure PII handling for user input, tool results, and output."""
    return [
        # Partially mask detected credit card numbers.
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_tool_results=True,
            apply_to_output=True,
        ),
        # Replace detected email addresses with redaction placeholders.
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
            apply_to_tool_results=True,
            apply_to_output=True,
        ),
    ]


def main() -> None:
    """Initialize the customer service agent and run a sample lookup."""
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "Set GROQ_API_KEY in your environment or .env file."
        )

    chat_model = ChatGroq(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
    )

    customer_service_agent = create_agent(
        model=chat_model,
        tools=[get_customer_info],
        system_prompt=SYSTEM_PROMPT,
        middleware=create_pii_middleware(),
    )

    customer_question = "Give me information about customer Krishna."

    agent_result = customer_service_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": customer_question,
                }
            ]
        }
    )

    final_response = agent_result["messages"][-1]

    print("\n=== CUSTOMER INFORMATION ===")
    print(final_response.content)


# Importing this module does not initialize the model or make API calls.
if __name__ == "__main__":
    main()
