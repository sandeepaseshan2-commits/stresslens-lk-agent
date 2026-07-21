"""Search the FAISS index using a research question."""

from pathlib import Path
from typing import Any
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_DIRECTORY = PROJECT_ROOT / "vector_store"
INDEX_PATH = VECTOR_DIRECTORY / "academic_stress.index"
CHUNKS_PATH = VECTOR_DIRECTORY / "chunks.json"
CONFIG_PATH = VECTOR_DIRECTORY / "config.json"


def load_search_resources() -> tuple[
    Any,
    list[dict[str, Any]],
    SentenceTransformer
]:
    """Load the index, chunk records and embedding model."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. "
            "Run: python -m rag.build_index"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "chunks.json not found. "
            "Run: python -m rag.build_index"
        )

    index = faiss.read_index(str(INDEX_PATH))

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    config = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )

    model = SentenceTransformer(
        config["embedding_model"]
    )

    return index, chunks, model


def search_documents(
    query: str,
    top_k: int = 5
) -> list[dict[str, Any]]:
    """Return the most relevant chunks for a question."""

    if not query.strip():
        raise ValueError("The research question cannot be empty.")

    index, chunks, model = load_search_resources()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    number_of_results = min(top_k, index.ntotal)

    scores, positions = index.search(
        query_embedding,
        number_of_results
    )

    results: list[dict[str, Any]] = []

    for score, position in zip(
        scores[0],
        positions[0]
    ):
        if position < 0:
            continue

        chunk = chunks[int(position)].copy()
        chunk["similarity_score"] = float(score)

        results.append(chunk)

    return results


def main() -> None:
    """Ask the user for a question and print five results."""

    print("StressLens LK — Research Evidence Search")
    print()

    question = input(
        "Enter your academic stress research question: "
    ).strip()

    try:
        results = search_documents(
            query=question,
            top_k=5
        )

        print()
        print("=" * 70)
        print(f"Top {len(results)} relevant chunks")
        print("=" * 70)

        for rank, result in enumerate(results, start=1):
            print()
            print(f"RESULT {rank}")
            print(f"Title: {result['title']}")
            print(f"Authors: {result['authors']}")
            print(f"Year: {result['year']}")
            print(f"Topic: {result['topic']}")
            print(f"Source: {result['source']}")
            print(
                "Similarity score: "
                f"{result['similarity_score']:.4f}"
            )
            print()
            print(result["text"][:1000])
            print()
            print("-" * 70)

    except Exception as error:
        print()
        print(f"Search failed: {error}")


if __name__ == "__main__":
    main()