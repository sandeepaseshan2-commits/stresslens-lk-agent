"""Streamlit interface for the StressLens LK multi-agent system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="StressLens LK",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Environment and secrets
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

# This loads the local .env file during development.
load_dotenv(PROJECT_ROOT / ".env")


def configure_streamlit_secrets() -> None:
    """Use Streamlit secrets when the app is deployed."""

    if os.getenv("GROQ_API_KEY"):
        return

    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None

    if secret_key:
        os.environ["GROQ_API_KEY"] = str(secret_key)

    try:
        fast_model = st.secrets.get("GROQ_FAST_MODEL")
        strong_model = st.secrets.get("GROQ_STRONG_MODEL")
    except Exception:
        fast_model = None
        strong_model = None

    if fast_model:
        os.environ["GROQ_FAST_MODEL"] = str(fast_model)

    if strong_model:
        os.environ["GROQ_STRONG_MODEL"] = str(strong_model)


configure_streamlit_secrets()


# Import these only after configuring the API key.
from agents.orchestrator import run_workflow  # noqa: E402
from models.groq_client import (  # noqa: E402
    FAST_MODEL,
    STRONG_MODEL,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def create_friendly_error(error: Exception) -> str:
    """Convert a technical exception into a clear user message."""

    error_text = str(error).lower()

    if (
        "groq_api_key" in error_text
        or "api key" in error_text
        or "401" in error_text
        or "unauthorized" in error_text
    ):
        return (
            "The Groq API key is missing or invalid. "
            "Check the local .env file."
        )

    if (
        "429" in error_text
        or "rate limit" in error_text
        or "too many requests" in error_text
    ):
        return (
            "The model request limit was reached. "
            "Wait a short time and try again."
        )

    if (
        "faiss index" in error_text
        or "index not found" in error_text
    ):
        return (
            "The FAISS knowledge base was not found. "
            "Run: python -m rag.build_index"
        )

    if "no evidence" in error_text:
        return (
            "The retrieval agent could not find suitable evidence "
            "for this question."
        )

    if "cannot be empty" in error_text:
        return "Please enter a research question."

    return (
        "The workflow could not be completed. "
        "Open the technical details below for more information."
    )


def display_plan(state: dict[str, Any]) -> None:
    """Display the router decision and task plan."""

    st.subheader("1. Router and planner result")

    st.write(
        "**Task type:**",
        str(state.get("task_type", "Unknown")).replace("_", " ").title(),
    )

    plan = state.get("plan", [])

    if plan:
        for number, step in enumerate(plan, start=1):
            st.write(f"{number}. {step}")
    else:
        st.info("No plan was recorded.")


def display_final_answer(state: dict[str, Any]) -> None:
    """Display the final evidence-based answer."""

    st.subheader("2. Final research answer")

    final_answer = state.get("final_answer", "")

    if final_answer:
        st.markdown(final_answer)
    else:
        st.warning("No final answer was produced.")

    draft_answer = state.get("draft_answer", "")

    if draft_answer:
        with st.expander("View the original draft"):
            st.markdown(draft_answer)


def display_sources(state: dict[str, Any]) -> None:
    """Display the evidence retrieved from FAISS."""

    st.subheader("3. Retrieved research evidence")

    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        st.warning("No retrieved sources are available.")
        return

    st.caption(
        "The labels [S1], [S2] and similar labels in the answer "
        "match the source numbers displayed below."
    )

    for number, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown document")
        source = chunk.get("source", "Unknown source")
        score = float(chunk.get("similarity_score", 0.0))

        expander_title = (
            f"[S{number}] {title} — similarity {score:.4f}"
        )

        with st.expander(expander_title):
            st.write("**Authors:**", chunk.get("authors", "Unknown"))
            st.write("**Year:**", chunk.get("year", "Unknown"))
            st.write("**Topic:**", chunk.get("topic", "Unknown"))
            st.write("**Source file:**", source)
            st.write("**Chunk ID:**", chunk.get("chunk_id", "Unknown"))
            st.write("**Retrieved context:**")
            st.write(chunk.get("text", ""))


def display_critic(state: dict[str, Any]) -> None:
    """Display the critic agent's decision."""

    st.subheader("4. Critic agent result")

    critique = state.get("critique")

    if not critique:
        st.info("No critic result was recorded.")
        return

    status = critique.get("status", "Unknown")
    score = critique.get("score", 0)

    column_one, column_two = st.columns(2)

    with column_one:
        st.metric("Critic score", f"{score}/100")

    with column_two:
        st.metric(
            "Decision",
            str(status).replace("_", " ").title(),
        )

    issues = critique.get("issues", [])
    instructions = critique.get("revision_instructions", [])

    if issues:
        st.write("**Problems identified:**")

        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("The critic did not identify a major problem.")

    if instructions:
        st.write("**Revision instructions:**")

        for instruction in instructions:
            st.write(f"- {instruction}")


def display_agent_messages(state: dict[str, Any]) -> None:
    """Display structured agent-to-agent messages."""

    st.subheader("5. Agent communication")

    messages = state.get("messages", [])

    if not messages:
        st.info("No structured messages were recorded.")
        return

    st.caption(
        "Each message records its sender, receiver, message type, "
        "correlation ID, timestamp and payload."
    )

    for number, message in enumerate(messages, start=1):
        sender = message.get("sender", "Unknown")
        receiver = message.get("receiver", "Unknown")
        message_type = message.get("message_type", "Unknown")

        title = (
            f"Message {number}: {sender} → {receiver} "
            f"({message_type})"
        )

        with st.expander(title):
            st.json(message)


def display_workflow_details(state: dict[str, Any]) -> None:
    """Display request and latency details."""

    st.subheader("6. Workflow information")

    left_column, right_column = st.columns(2)

    with left_column:
        st.write(
            "**Request ID:**",
            state.get("request_id", "Unknown"),
        )

        st.write(
            "**Workflow status:**",
            state.get("status", "Unknown"),
        )

    with right_column:
        st.write(
            "**Structured messages:**",
            len(state.get("messages", [])),
        )

        st.write(
            "**Retrieved chunks:**",
            len(state.get("retrieved_chunks", [])),
        )

    with st.expander("View model and retrieval latency"):
        st.json(state.get("latencies", {}))


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if "research_question" not in st.session_state:
    st.session_state.research_question = ""


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("About StressLens LK")

    st.write(
        "An Agentic AI research support system for academic "
        "stress studies among Sri Lankan private campus "
        "undergraduates."
    )

    st.divider()

    st.write("**Fast model**")
    st.code(FAST_MODEL)

    st.write("**Strong model**")
    st.code(STRONG_MODEL)

    st.write("**Knowledge source**")
    st.write("20+ domain-specific research documents")

    st.write("**Vector search**")
    st.write("Sentence Transformers + FAISS")

    st.divider()

    if st.button("Clear current result", use_container_width=True):
        st.session_state.workflow_result = None
        st.session_state.research_question = ""


# ---------------------------------------------------------
# Main interface
# ---------------------------------------------------------

st.title("🎓 StressLens LK")

st.subheader(
    "Agentic Research Support for Academic Stress Studies"
)

st.write(
    "Ask a research question. The system will plan the task, "
    "retrieve evidence, create a draft, check it and return a "
    "source-supported answer."
)

st.info(
    "The system answers from the project knowledge base. "
    "It may state that evidence is insufficient when the documents "
    "do not support the question."
)

with st.form("research_question_form"):
    question = st.text_area(
        "Enter your academic stress research question",
        key="research_question",
        height=130,
        placeholder=(
            "Example: How does examination pressure contribute "
            "to academic stress among undergraduate students?"
        ),
    )

    submitted = st.form_submit_button(
        "Analyse research question",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Run workflow
# ---------------------------------------------------------

if submitted:
    if not question.strip():
        st.warning("Please enter a research question.")

    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "The Groq API key is missing. "
            "Add GROQ_API_KEY to the local .env file."
        )

    else:
        try:
            with st.spinner(
                "The agents are retrieving and checking evidence..."
            ):
                workflow_state = run_workflow(
                    question.strip()
                )

                # Store a normal JSON-compatible dictionary.
                st.session_state.workflow_result = (
                    workflow_state.model_dump(mode="json")
                )

            st.success("The multi-agent workflow completed successfully.")

        except Exception as error:
            st.session_state.workflow_result = None

            st.error(create_friendly_error(error))

            with st.expander("Technical error details"):
                st.code(str(error))


# ---------------------------------------------------------
# Display saved result
# ---------------------------------------------------------

state = st.session_state.workflow_result

if state:
    st.divider()

    display_plan(state)
    st.divider()

    display_final_answer(state)
    st.divider()

    display_sources(state)
    st.divider()

    display_critic(state)
    st.divider()

    display_agent_messages(state)
    st.divider()

    display_workflow_details(state)