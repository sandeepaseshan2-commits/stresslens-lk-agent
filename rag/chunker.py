"""Divide long research documents into smaller overlapping chunks."""

from typing import Any


def chunk_document(
    document: dict[str, Any],
    chunk_size: int = 180,
    overlap: int = 40
) -> list[dict[str, Any]]:
    """Split one document using word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be zero or less than chunk_size."
        )

    words = document["text"].split()
    step_size = chunk_size - overlap

    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for start in range(0, len(words), step_size):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        if len(chunk_words) < 30:
            break

        chunks.append(
            {
                "chunk_id": (
                    f"{document['document_id']}"
                    f"_CHUNK_{chunk_number:03d}"
                ),
                "document_id": document["document_id"],
                "title": document["title"],
                "authors": document["authors"],
                "year": document["year"],
                "topic": document["topic"],
                "source": document["source"],
                "text": " ".join(chunk_words),
            }
        )

        chunk_number += 1

        if end == len(words):
            break

    return chunks


def create_all_chunks(
    documents: list[dict[str, Any]],
    chunk_size: int = 180,
    overlap: int = 40
) -> list[dict[str, Any]]:
    """Create chunks for every document."""

    all_chunks: list[dict[str, Any]] = []

    for document in documents:
        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        all_chunks.extend(document_chunks)

    return all_chunks