"""Validated message structures shared by all agents."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


TaskType = Literal[
    "literature_summary",
    "factor_comparison",
    "coping_strategy",
    "research_gap",
    "questionnaire_support",
]

MessageType = Literal[
    "task_request",
    "plan_response",
    "retrieval_request",
    "evidence_response",
    "synthesis_request",
    "draft_response",
    "critique_request",
    "critique_response",
    "revision_request",
    "final_response",
]


def create_request_id() -> str:
    """Create a unique ID for one user request."""

    return f"REQ-{uuid4().hex[:8].upper()}"


class AgentMessage(BaseModel):
    """One structured message sent between two agents."""

    protocol_version: str = "1.0"
    correlation_id: str
    sender: str
    receiver: str
    message_type: MessageType
    payload: dict[str, Any]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )


class RouterDecision(BaseModel):
    """Validated output from the router agent."""

    task_type: TaskType
    reason: str
    plan: list[str] = Field(
        min_length=2,
        max_length=5,
    )


class RetrievedChunk(BaseModel):
    """One piece of evidence returned by FAISS."""

    chunk_id: str
    document_id: str
    title: str
    authors: str
    year: str
    topic: str
    source: str
    text: str
    similarity_score: float


class CritiqueResult(BaseModel):
    """Validated output from the critic agent."""

    status: Literal[
        "approved",
        "revision_required",
    ]

    score: int = Field(
        ge=0,
        le=100,
    )

    issues: list[str] = Field(
        default_factory=list
    )

    revision_instructions: list[str] = Field(
        default_factory=list
    )


class AgentState(BaseModel):
    """Complete shared state of one workflow."""

    request_id: str = Field(
        default_factory=create_request_id
    )

    user_query: str
    status: str = "started"

    task_type: TaskType | None = None
    plan: list[str] = Field(
        default_factory=list
    )

    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list
    )

    draft_answer: str = ""
    critique: CritiqueResult | None = None
    final_answer: str = ""

    messages: list[AgentMessage] = Field(
        default_factory=list
    )

    latencies: dict[str, float] = Field(
        default_factory=dict
    )