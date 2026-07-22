"""Review a draft answer and request corrections when needed."""

from models.groq_client import (
    STRONG_MODEL,
    GroqModelClient,
    ModelCallResult,
)
from schemas.agent_state import (
    AgentMessage,
    CritiqueResult,
    RetrievedChunk,
)
from utils.json_utils import extract_json_object


def create_source_summary(
    chunks: list[RetrievedChunk],
) -> str:
    """Create a short source list for the critic."""

    source_lines: list[str] = []

    for number, chunk in enumerate(
        chunks,
        start=1,
    ):
        source_lines.append(
            f"[S{number}] {chunk.title}: "
            f"{chunk.text[:700]}"
        )

    return "\n\n".join(source_lines)


def review_draft(
    query: str,
    draft_answer: str,
    chunks: list[RetrievedChunk],
    client: GroqModelClient,
    request_id: str,
) -> tuple[
    CritiqueResult,
    AgentMessage,
    ModelCallResult,
]:
    """Check evidence use and answer quality."""

    evidence_summary = create_source_summary(
        chunks
    )

    system_prompt = """
You are the Critic Agent for an academic research support system.

Check the draft using these rules:

1. It must directly answer the research question.
2. Important factual claims must be supported by [S1], [S2] and
   similar source labels.
3. It must not contain unsupported statistics or invented findings.
4. It must use only the supplied evidence.
5. It must clearly state when evidence is insufficient.
6. It should be understandable and logically organised.

Return only JSON in this exact structure:

{
  "status": "approved",
  "score": 90,
  "issues": [],
  "revision_instructions": []
}

Use "revision_required" when corrections are needed.
""".strip()

    user_prompt = f"""
Research question:
{query}

Available evidence:
{evidence_summary}

Draft answer:
{draft_answer}

Review the draft and return the JSON result.
""".strip()

    result = client.generate(
        model=STRONG_MODEL,
        temperature=0.0,
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

    try:
        parsed_output = extract_json_object(
            result.text
        )

        critique = CritiqueResult.model_validate(
            parsed_output
        )

    except Exception:
        critique = CritiqueResult(
            status="revision_required",
            score=60,
            issues=[
                (
                    "The critic response could not be "
                    "validated as structured JSON."
                )
            ],
            revision_instructions=[
                (
                    "Ensure every important claim has a "
                    "source label and remove unsupported claims."
                )
            ],
        )

    response_message = AgentMessage(
        correlation_id=request_id,
        sender="critic_agent",
        receiver="orchestrator",
        message_type="critique_response",
        payload={
            "model": result.returned_model,
            "status": critique.status,
            "score": critique.score,
            "issues": critique.issues,
            "revision_instructions": (
                critique.revision_instructions
            ),
        },
    )

    return critique, response_message, result