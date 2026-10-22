"""Streamlit interface for the telecom support assistant."""

import logging

import streamlit as st
from langchain_core.runnables import Runnable

from rag_chain import build_chain


logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "My calls keep dropping — what should I do?",
    "How do I activate international roaming?",
    "Why is my bill higher than usual this month?",
    "My phone shows SIM not detected after a restart",
    "How do I enable Wi-Fi calling?",
    "I was charged for roaming but had a bundle active",
    "How do I unlock my phone for another network?",
]


@st.cache_resource
def get_chain() -> Runnable:
    """Reuse the RAG chain across Streamlit reruns."""
    return build_chain()


def initialize_chat_state() -> None:
    """Initialize state before the interface accesses it."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_sidebar() -> None:
    """Display sample questions and conversation controls."""
    with st.sidebar:
        st.title("📡 Telecom Support")
        st.caption("Answers from FAQs, resolved tickets, and guides.")
        st.divider()

        st.markdown("**Sample questions**")
        st.caption("Click a question to send it.")

        for question_index, sample_question in enumerate(
            SAMPLE_QUESTIONS
        ):
            if st.button(
                sample_question,
                key=f"sample_question_{question_index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = sample_question

        st.divider()

        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()


def render_chat_history() -> None:
    """Display messages retained for the current browser session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(
                message["content"].replace("$", r"\$")
            )


def answer_question(question: str) -> None:
    """Stream an answer and save the completed exchange."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question.replace("$", r"\$"))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching support information…"):
                rag_chain = get_chain()

            answer = st.write_stream(
                rag_chain.stream(question)
            )

        except Exception:
            logger.exception("Failed to answer a support question")

            answer = (
                "I couldn't complete the request. Please check that "
                "the data has been ingested and the service is available."
            )
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


def main() -> None:
    """Render the application and handle submitted questions."""
    st.set_page_config(
        page_title="Telecom Support Chat",
        page_icon="📡",
        layout="centered",
    )

    initialize_chat_state()
    render_sidebar()

    st.title("Customer Care Assistant")
    st.caption(
        "Ask about connectivity, billing, SIM issues, roaming, and more."
    )

    render_chat_history()

    typed_question = st.chat_input("Describe your issue…")

    # Sidebar questions take priority when a sample button was clicked.
    question = st.session_state.pending_question or typed_question
    st.session_state.pending_question = None

    if question and question.strip():
        answer_question(question.strip())


if __name__ == "__main__":
    main()
