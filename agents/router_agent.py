"""Classify the research request and create a short plan."""

from models.groq_client import (
    FAST_MODEL,
    GroqModelClient,
    ModelCallResult,
)
from schemas.agent_state import (
    AgentMessage,
    RouterDecision,
)
from utils.json_utils import extract_json_object


def fallback_decision(
    query: str,
) -> RouterDecision:
    """Create a basic plan if model JSON cannot be parsed."""

    query_lower = query.lower()

    if (
        "compare" in query_lower
        or "difference" in query_lower
        or "versus" in query_lower
    ):
        task_type = "factor_comparison"

    elif (
        "coping" in query_lower
        or "manage stress" in query_lower
    ):
        task_type = "coping_strategy"

    elif (
        "research gap" in query_lower
        or "gap" in query_lower
    ):
        task_type = "research_gap"

    elif (
        "questionnaire" in query_lower
        or "survey question" in query_lower
    ):
        task_type = "questionnaire_support"

    else:
        task_type = "literature_summary"

    return RouterDecision(
        task_type=task_type,
        reason=(
            "Fallback keyword classification was used."
        ),
        plan=[
            "Retrieve relevant evidence from the knowledge base",
            "Analyse the retrieved research findings",
            "Prepare an answer supported by source citations",
        ],
    )


def route_and_plan(
    query: str,
    client: GroqModelClient,
    request_id: str,
) -> tuple[
    RouterDecision,
    AgentMessage,
    ModelCallResult,
]:
    """Classify a question using the fast model."""

    system_prompt = """
You are the Router and Planner Agent for StressLens LK.

Classify the user's research question into exactly one task type:

1. literature_summary
2. factor_comparison
3. coping_strategy
4. research_gap
5. questionnaire_support

Create a short plan containing 2 to 5 steps.

Return only a JSON object in this exact structure:

{
  "task_type": "literature_summary",
  "reason": "Short reason for the decision",
  "plan": [
    "First step",
    "Second step",
    "Third step"
  ]
}
""".strip()

    result = client.generate(
        model=FAST_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )

    try:
        parsed_output = extract_json_object(
            result.text
        )

        decision = RouterDecision.model_validate(
            parsed_output
        )

    except Exception:
        decision = fallback_decision(query)

    response_message = AgentMessage(
        correlation_id=request_id,
        sender="router_agent",
        receiver="orchestrator",
        message_type="plan_response",
        payload={
            "model": result.returned_model,
            "task_type": decision.task_type,
            "reason": decision.reason,
            "plan": decision.plan,
        },
    )

    return decision, response_message, result