"""Command-line entry point for the telecom support assistant."""

import logging

from rag_chain import build_chain


logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"quit", "exit", "q"}


def main() -> None:
    """Initialize the RAG chain and stream answers in the terminal."""
    print("=== Telecom Customer Care Chatbot ===")
    print("Type a question, or type 'quit' to exit.\n")

    rag_chain = build_chain()

    try:
        while True:
            question = input("Customer: ").strip()

            if question.casefold() in EXIT_COMMANDS:
                break

            if not question:
                continue

            print("\nAssistant: ", end="", flush=True)

            try:
                for response_chunk in rag_chain.stream(question):
                    print(response_chunk, end="", flush=True)

            except Exception:
                logger.exception("Failed to generate a support response")
                print(
                    "\nThe request could not be completed. "
                    "Please try again."
                )

            print("\n")

    except (KeyboardInterrupt, EOFError):
        print()

    print("Goodbye!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
