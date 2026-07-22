"""Generate and revise evidence-based research answers."""

from models.groq_client import (
    STRONG_MODEL,
    GroqModelClient,
    ModelCallResult,
)
from schemas.agent_state import (
    AgentMessage,
    CritiqueResult,
    RetrievedChunk,
    RouterDecision,
)


def format_evidence(
    chunks: list[RetrievedChunk],
) -> str:
    """Prepare retrieved chunks for the language model."""

    evidence_sections: list[str] = []

    for number, chunk in enumerate(
        chunks,
        start=1,
    ):
        evidence_sections.append(
            f"""
[S{number}]
Title: {chunk.title}
Authors: {chunk.authors}
Year: {chunk.year}
Source file: {chunk.source}
Similarity score: {chunk.similarity_score:.4f}

Evidence:
{chunk.text}
""".strip()
        )

    return "\n\n".join(evidence_sections)


def generate_draft(
    query: str,
    decision: RouterDecision,
    chunks: list[RetrievedChunk],
    client: GroqModelClient,
    request_id: str,
) -> tuple[
    str,
    AgentMessage,
    ModelCallResult,
]:
    """Produce the first research answer."""

    evidence_context = format_evidence(chunks)

    system_prompt = """
You are the Research Synthesis Agent for StressLens LK.

Your job is to answer the researcher's question using only the
retrieved evidence supplied to you.

Rules:

1. Do not use unsupported outside knowledge.
2. Cite evidence using [S1], [S2], [S3] and similar labels.
3. Do not invent paper titles, authors, results or statistics.
4. Clearly state when the evidence is insufficient.
5. Compare studies when the question asks for comparison.
6. Use clear academic English that a university student can explain.
7. Do not create a separate reference list because source details
   will be displayed by the application.
""".strip()

    user_prompt = f"""
Research question:
{query}

Task type:
{decision.task_type}

Planned steps:
{decision.plan}

Retrieved evidence:
{evidence_context}

Prepare a clear evidence-based answer.
""".strip()

    result = client.generate(
        model=STRONG_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    response_message = AgentMessage(
        correlation_id=request_id,
        sender="synthesis_agent",
        receiver="orchestrator",
        message_type="draft_response",
        payload={
            "model": result.returned_model,
            "used_chunk_ids": [
                chunk.chunk_id
                for chunk in chunks
            ],
            "draft_answer": result.text,
        },
    )

    return result.text, response_message, result


def revise_draft(
    query: str,
    original_draft: str,
    critique: CritiqueResult,
    chunks: list[RetrievedChunk],
    client: GroqModelClient,
    request_id: str,
) -> tuple[
    str,
    AgentMessage,
    ModelCallResult,
]:
    """Revise the answer using critic feedback."""

    evidence_context = format_evidence(chunks)

    system_prompt = """
You are revising an academic research answer.

Use only the supplied evidence.

Correct every problem identified by the Critic Agent.

Every major factual claim must contain a source label such as [S1].
Do not invent new facts or sources.
""".strip()

    user_prompt = f"""
Research question:
{query}

Original draft:
{original_draft}

Critic issues:
{critique.issues}

Revision instructions:
{critique.revision_instructions}

Available evidence:
{evidence_context}

Return only the corrected final answer.
""".strip()

    result = client.generate(
        model=STRONG_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    response_message = AgentMessage(
        correlation_id=request_id,
        sender="synthesis_agent",
        receiver="orchestrator",
        message_type="final_response",
        payload={
            "model": result.returned_model,
            "final_answer": result.text,
            "revision_completed": True,
        },
    )

    return result.text, response_message, result