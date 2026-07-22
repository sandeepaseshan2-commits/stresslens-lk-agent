"""Retrieve relevant academic evidence from the FAISS index."""

from time import perf_counter

from rag.search import search_documents
from schemas.agent_state import (
    AgentMessage,
    RetrievedChunk,
)


def retrieve_evidence(
    query: str,
    request_id: str,
    top_k: int = 5,
) -> tuple[
    list[RetrievedChunk],
    AgentMessage,
    float,
]:
    """Find relevant and reasonably diverse chunks."""

    start_time = perf_counter()

    # Retrieve extra results so that we can reduce
    # repeated results from the same document.
    raw_results = search_documents(
        query=query,
        top_k=max(top_k * 2, 10),
    )

    selected_results: list[dict] = []
    source_counts: dict[str, int] = {}

    for result in raw_results:
        source = str(result.get("source", ""))

        # Keep no more than two chunks from one paper.
        if source_counts.get(source, 0) >= 2:
            continue

        selected_results.append(result)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(selected_results) == top_k:
            break

    # Fill any missing positions when fewer than
    # five diverse results were available.
    selected_chunk_ids = {
        str(item.get("chunk_id"))
        for item in selected_results
    }

    for result in raw_results:
        if len(selected_results) == top_k:
            break

        chunk_id = str(
            result.get("chunk_id", "")
        )

        if chunk_id not in selected_chunk_ids:
            selected_results.append(result)
            selected_chunk_ids.add(chunk_id)

    chunks: list[RetrievedChunk] = []

    for result in selected_results:
        chunks.append(
            RetrievedChunk(
                chunk_id=str(
                    result.get("chunk_id", "")
                ),
                document_id=str(
                    result.get("document_id", "")
                ),
                title=str(
                    result.get("title", "Unknown")
                ),
                authors=str(
                    result.get("authors", "Unknown")
                ),
                year=str(
                    result.get("year", "Unknown")
                ),
                topic=str(
                    result.get(
                        "topic",
                        "Academic stress",
                    )
                ),
                source=str(
                    result.get("source", "Unknown")
                ),
                text=str(
                    result.get("text", "")
                ),
                similarity_score=float(
                    result.get(
                        "similarity_score",
                        0.0,
                    )
                ),
            )
        )

    latency = perf_counter() - start_time

    response_message = AgentMessage(
        correlation_id=request_id,
        sender="retrieval_agent",
        receiver="orchestrator",
        message_type="evidence_response",
        payload={
            "tool": "FAISS semantic search",
            "number_of_chunks": len(chunks),
            "chunk_ids": [
                chunk.chunk_id
                for chunk in chunks
            ],
            "sources": [
                chunk.source
                for chunk in chunks
            ],
        },
    )

    return chunks, response_message, latency