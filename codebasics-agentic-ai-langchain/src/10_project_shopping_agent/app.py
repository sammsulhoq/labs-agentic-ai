"""Streamlit interface for the AI shopping assistant."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from shopping_agent import run_shopping_agent


logger = logging.getLogger(__name__)


def initialize_chat_state() -> None:
    """Initialize conversation history before any widget accesses it."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_message(role: str, content: str) -> None:
    """Render a chat message while displaying dollar signs literally."""
    with st.chat_message(role):
        st.markdown(content.replace("$", r"\$"))


def display_chat_history() -> None:
    """Render the current session's conversation."""
    for message in st.session_state.messages:
        display_message(
            role=message["role"],
            content=message["content"],
        )


def process_user_message(
    agent_prompt: str,
    display_text: str | None = None,
) -> None:
    """Submit one message and append the assistant's final answer."""
    visible_prompt = (
        display_text if display_text is not None else agent_prompt
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": agent_prompt,
        }
    )
    user_message_index = len(st.session_state.messages) - 1

    display_message("user", visible_prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching and preparing your answer…"):
                answer = run_shopping_agent(
                    st.session_state.messages
                )

            st.markdown(answer.replace("$", r"\$"))

    except Exception:
        # Log technical details on the server rather than displaying them.
        # Do not automatically retry: a checkout may already have completed.
        logger.exception("Shopping assistant request failed")

        answer = (
            "I couldn't complete the request. If you were placing an "
            "order, check the orders table before trying again."
        )
        display_message("assistant", answer)

    finally:
        # Keep the friendly image label in history rather than a temporary
        # file path that will no longer exist after this request.
        st.session_state.messages[user_message_index]["content"] = (
            visible_prompt
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


def process_uploaded_image(
    image_bytes: bytes,
    uploaded_filename: str,
) -> None:
    """Run image search and clean up its temporary file afterward."""
    image_suffix = Path(uploaded_filename).suffix.lower() or ".jpg"

    # The directory and image are removed even if processing fails.
    with TemporaryDirectory(prefix="shopping_image_") as temp_directory:
        temporary_image_path = (
            Path(temp_directory) / f"product{image_suffix}"
        )
        temporary_image_path.write_bytes(image_bytes)

        image_prompt = (
            "I uploaded a product image. Analyze it and find similar "
            "products in the store. This is a search request, not an "
            "order confirmation.\n"
            f"Uploaded image path: {temporary_image_path}"
        )

        process_user_message(
            agent_prompt=image_prompt,
            display_text=(
                f"Find products similar to my uploaded image: "
                f"{uploaded_filename}"
            ),
        )


def render_image_sidebar() -> tuple[bytes, str] | None:
    """Display the image uploader and return a submitted image."""
    with st.sidebar:
        st.header("Shop by Image")
        st.caption(
            "Upload a product photo to find similar items in our store."
        )

        uploaded_image = st.file_uploader(
            "Upload product image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_image is not None:
            st.image(
                uploaded_image,
                use_container_width=True,
            )

        image_search_requested = st.button(
            "Find similar products",
            disabled=uploaded_image is None,
            use_container_width=True,
        )

        if image_search_requested and uploaded_image is not None:
            return (
                uploaded_image.getvalue(),
                uploaded_image.name,
            )

    return None


def main() -> None:
    """Render the application and process text or image requests."""
    st.set_page_config(
        page_title="AI Shopping Assistant",
        page_icon="🛒",
        layout="wide",
    )

    initialize_chat_state()

    st.title("🛒 AI Shopping Assistant")
    st.caption(
        "Find products, compare customer ratings, and place demo orders."
    )

    submitted_image = render_image_sidebar()
    display_chat_history()

    if submitted_image is not None:
        image_bytes, uploaded_filename = submitted_image

        process_uploaded_image(
            image_bytes=image_bytes,
            uploaded_filename=uploaded_filename,
        )
        st.rerun()

    user_question = st.chat_input(
        "e.g. Find organic honey under $15 with a 4+ rating"
    )

    if user_question and user_question.strip():
        process_user_message(
            agent_prompt=user_question.strip(),
        )
        st.rerun()


if __name__ == "__main__":
    main()
