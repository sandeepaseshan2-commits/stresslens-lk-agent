"""Control the complete multi-agent research workflow."""

from pathlib import Path

from agents.critic_agent import review_draft
from agents.retrieval_agent import retrieve_evidence
from agents.router_agent import route_and_plan
from agents.synthesis_agent import (
    generate_draft,
    revise_draft,
)
from models.groq_client import (
    FAST_MODEL,
    STRONG_MODEL,
    GroqModelClient,
)
from schemas.agent_state import (
    AgentMessage,
    AgentState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACE_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "latest_agent_trace.json"
)


def add_request_message(
    state: AgentState,
    receiver: str,
    message_type: str,
    payload: dict,
) -> None:
    """Record a structured request from the orchestrator."""

    state.messages.append(
        AgentMessage(
            correlation_id=state.request_id,
            sender="orchestrator",
            receiver=receiver,
            message_type=message_type,
            payload=payload,
        )
    )


def save_trace(state: AgentState) -> None:
    """Save all structured agent messages for inspection."""

    TRACE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRACE_FILE.write_text(
        state.model_dump_json(indent=2),
        encoding="utf-8",
    )


def run_workflow(
    query: str,
) -> AgentState:
    """Run router, retrieval, synthesis and criticism."""

    if not query.strip():
        raise ValueError(
            "Please enter a research question."
        )

    state = AgentState(
        user_query=query.strip()
    )

    client = GroqModelClient()

    print()
    print("Question received")

    # -------------------------------------------------
    # 1. Router and planner
    # -------------------------------------------------

    add_request_message(
        state=state,
        receiver="router_agent",
        message_type="task_request",
        payload={
            "query": state.user_query,
            "assigned_model": FAST_MODEL,
        },
    )

    decision, router_message, router_result = (
        route_and_plan(
            query=state.user_query,
            client=client,
            request_id=state.request_id,
        )
    )

    state.task_type = decision.task_type
    state.plan = decision.plan
    state.messages.append(router_message)

    state.latencies["router_model"] = round(
        router_result.latency_seconds,
        4,
    )

    print("Router classified task")

    # -------------------------------------------------
    # 2. Retrieval agent
    # -------------------------------------------------

    add_request_message(
        state=state,
        receiver="retrieval_agent",
        message_type="retrieval_request",
        payload={
            "query": state.user_query,
            "top_k": 5,
            "tool": "FAISS",
        },
    )

    chunks, retrieval_message, retrieval_latency = (
        retrieve_evidence(
            query=state.user_query,
            request_id=state.request_id,
            top_k=5,
        )
    )

    if not chunks:
        raise RuntimeError(
            "The retrieval agent found no evidence."
        )

    state.retrieved_chunks = chunks
    state.messages.append(retrieval_message)

    state.latencies["retrieval"] = round(
        retrieval_latency,
        4,
    )

    print("Retrieval found evidence")

    # -------------------------------------------------
    # 3. Synthesis agent
    # -------------------------------------------------

    add_request_message(
        state=state,
        receiver="synthesis_agent",
        message_type="synthesis_request",
        payload={
            "query": state.user_query,
            "task_type": state.task_type,
            "plan": state.plan,
            "chunk_ids": [
                chunk.chunk_id
                for chunk in chunks
            ],
            "assigned_model": STRONG_MODEL,
        },
    )

    draft, draft_message, synthesis_result = (
        generate_draft(
            query=state.user_query,
            decision=decision,
            chunks=chunks,
            client=client,
            request_id=state.request_id,
        )
    )

    state.draft_answer = draft
    state.messages.append(draft_message)

    state.latencies["synthesis_model"] = round(
        synthesis_result.latency_seconds,
        4,
    )

    print("Synthesis created draft")

    # -------------------------------------------------
    # 4. Critic agent
    # -------------------------------------------------

    add_request_message(
        state=state,
        receiver="critic_agent",
        message_type="critique_request",
        payload={
            "query": state.user_query,
            "draft_answer": state.draft_answer,
            "assigned_model": STRONG_MODEL,
        },
    )

    critique, critique_message, critic_result = (
        review_draft(
            query=state.user_query,
            draft_answer=state.draft_answer,
            chunks=chunks,
            client=client,
            request_id=state.request_id,
        )
    )

    state.critique = critique
    state.messages.append(critique_message)

    state.latencies["critic_model"] = round(
        critic_result.latency_seconds,
        4,
    )

    print("Critic checked draft")

    # -------------------------------------------------
    # 5. Reflection and revision
    # -------------------------------------------------

    if critique.status == "revision_required":
        add_request_message(
            state=state,
            receiver="synthesis_agent",
            message_type="revision_request",
            payload={
                "original_draft": (
                    state.draft_answer
                ),
                "issues": critique.issues,
                "revision_instructions": (
                    critique.revision_instructions
                ),
                "assigned_model": STRONG_MODEL,
            },
        )

        final_answer, final_message, revision_result = (
            revise_draft(
                query=state.user_query,
                original_draft=state.draft_answer,
                critique=critique,
                chunks=chunks,
                client=client,
                request_id=state.request_id,
            )
        )

        state.final_answer = final_answer
        state.messages.append(final_message)

        state.latencies["revision_model"] = round(
            revision_result.latency_seconds,
            4,
        )

    else:
        state.final_answer = state.draft_answer

        state.messages.append(
            AgentMessage(
                correlation_id=state.request_id,
                sender="orchestrator",
                receiver="user",
                message_type="final_response",
                payload={
                    "final_answer": (
                        state.final_answer
                    ),
                    "revision_completed": False,
                },
            )
        )

    state.status = "completed"

    save_trace(state)

    print("Final answer produced")

    return state


def main() -> None:
    """Run the workflow through the terminal."""

    print("=" * 70)
    print("StressLens LK")
    print("Multi-Agent Research Support Workflow")
    print("=" * 70)
    print()

    question = input(
        "Enter your academic stress research question: "
    ).strip()

    try:
        state = run_workflow(question)

    except Exception as error:
        print()
        print(f"Workflow failed: {error}")
        return

    print()
    print("=" * 70)
    print("ROUTER DECISION")
    print("=" * 70)
    print(f"Task type: {state.task_type}")

    for step_number, step in enumerate(
        state.plan,
        start=1,
    ):
        print(f"{step_number}. {step}")

    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for number, chunk in enumerate(
        state.retrieved_chunks,
        start=1,
    ):
        print(
            f"{number}. {chunk.title} "
            f"({chunk.source})"
        )
        print(
            f"   Similarity: "
            f"{chunk.similarity_score:.4f}"
        )

    print()
    print("=" * 70)
    print("CRITIC RESULT")
    print("=" * 70)

    if state.critique:
        print(
            f"Status: {state.critique.status}"
        )
        print(f"Score: {state.critique.score}")

        if state.critique.issues:
            for issue in state.critique.issues:
                print(f"- {issue}")

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(state.final_answer)

    print()
    print("=" * 70)
    print("WORKFLOW INFORMATION")
    print("=" * 70)
    print(f"Request ID: {state.request_id}")
    print(
        f"Structured messages: "
        f"{len(state.messages)}"
    )
    print(f"Latencies: {state.latencies}")
    print(f"Trace saved to: {TRACE_FILE}")


if __name__ == "__main__":
    main()