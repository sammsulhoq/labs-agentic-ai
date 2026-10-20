"""Query Google and Groq models using LangChain."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 256

QUESTION = "How many states are there in India?"
SYSTEM_PROMPT = "You are a helpful assistant who answers in short sentences."


def main() -> None:
    """Initialize the models and run both prompting examples."""
    load_dotenv()

    models = {
        "Google": ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
        "Groq": ChatGroq(
            model="qewn/qewn3-32b",
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    }

    for provider, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Provider: {provider}")
        print("=" * 50)

        # Example 1: Pass the question directly as a string.
        response = model.invoke(QUESTION)

        print("\n1. Plain string prompt")
        print(response.text)

        # Example 2: Set behavior with a system message,
        # then provide the question as a human message.
        response_with_system = model.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", QUESTION),
            ]
        )

        print("\n2. System and human messages")
        print(response_with_system.text)


if __name__ == "__main__":
    main()
