"""Manually evaluate the relevance of retrieved FAISS chunks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rag.search import search_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_questions.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_evaluation.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_summary.md"
)

TOP_K = 5


def ask_relevance() -> str:
    """Ask the evaluator to mark one result as relevant or irrelevant."""

    while True:
        answer = input(
            "Is this result relevant? Enter y or n: "
        ).strip().lower()

        if answer == "y":
            return "Yes"

        if answer == "n":
            return "No"

        print("Please enter only y or n.")


def load_questions() -> list[dict[str, str]]:
    """Load the five evaluation questions."""

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Question file not found: {QUESTIONS_PATH}"
        )

    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(questions, list):
        raise ValueError(
            "retrieval_questions.json must contain a list."
        )

    return questions


def write_results(
    records: list[dict[str, Any]],
) -> None:
    """Save all manual evaluation decisions to CSV."""

    fieldnames = [
        "question_id",
        "question",
        "rank",
        "chunk_id",
        "title",
        "authors",
        "year",
        "source",
        "similarity_score",
        "relevant",
        "reason",
        "retrieved_context",
    ]

    with RESULTS_PATH.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def write_summary(
    question_summaries: list[dict[str, Any]],
    total_relevant: int,
    total_results: int,
) -> None:
    """Create a short Markdown evaluation summary."""

    overall_precision = (
        total_relevant / total_results
        if total_results
        else 0.0
    )

    lines = [
        "# Retrieval Evaluation Summary",
        "",
        "The FAISS retrieval pipeline was evaluated using "
        "five domain-specific research questions.",
        "",
        "| Question | Relevant results | Retrieved results | Precision@5 |",
        "|---|---:|---:|---:|",
    ]

    for summary in question_summaries:
        lines.append(
            f"| {summary['question_id']} "
            f"| {summary['relevant_count']} "
            f"| {summary['result_count']} "
            f"| {summary['precision']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"**Overall relevant results:** "
            f"{total_relevant}/{total_results}",
            "",
            f"**Overall retrieval precision:** "
            f"{overall_precision:.2f}",
            "",
            "Detailed retrieved contexts and manual relevance "
            "decisions are available in "
            "`evaluation/retrieval_evaluation.csv`.",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run the five-question manual retrieval evaluation."""

    questions = load_questions()

    all_records: list[dict[str, Any]] = []
    question_summaries: list[dict[str, Any]] = []

    total_relevant = 0
    total_results = 0

    print("=" * 75)
    print("StressLens LK — Manual Retrieval Evaluation")
    print("=" * 75)

    for question_number, question_data in enumerate(
        questions,
        start=1,
    ):
        question_id = question_data["question_id"]
        question = question_data["question"]

        print()
        print("=" * 75)
        print(
            f"QUESTION {question_number}/{len(questions)} "
            f"— {question_id}"
        )
        print(question)
        print("=" * 75)

        results = search_documents(
            query=question,
            top_k=TOP_K,
        )

        relevant_count = 0

        for rank, result in enumerate(
            results,
            start=1,
        ):
            text = str(result.get("text", ""))

            print()
            print("-" * 75)
            print(f"RESULT {rank}")
            print(f"Title: {result.get('title', 'Unknown')}")
            print(f"Source: {result.get('source', 'Unknown')}")
            print(
                "Similarity score: "
                f"{float(result.get('similarity_score', 0.0)):.4f}"
            )
            print()
            print("Retrieved context:")
            print(text[:1200])
            print()

            relevance = ask_relevance()

            reason = input(
                "Briefly explain why it is relevant or irrelevant: "
            ).strip()

            if not reason:
                reason = "No explanation entered."

            if relevance == "Yes":
                relevant_count += 1

            all_records.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "rank": rank,
                    "chunk_id": result.get(
                        "chunk_id",
                        "Unknown",
                    ),
                    "title": result.get(
                        "title",
                        "Unknown",
                    ),
                    "authors": result.get(
                        "authors",
                        "Unknown",
                    ),
                    "year": result.get(
                        "year",
                        "Unknown",
                    ),
                    "source": result.get(
                        "source",
                        "Unknown",
                    ),
                    "similarity_score": round(
                        float(
                            result.get(
                                "similarity_score",
                                0.0,
                            )
                        ),
                        4,
                    ),
                    "relevant": relevance,
                    "reason": reason,
                    "retrieved_context": text,
                }
            )

        result_count = len(results)

        precision = (
            relevant_count / result_count
            if result_count
            else 0.0
        )

        question_summaries.append(
            {
                "question_id": question_id,
                "relevant_count": relevant_count,
                "result_count": result_count,
                "precision": precision,
            }
        )

        total_relevant += relevant_count
        total_results += result_count

        print()
        print(
            f"{question_id} Precision@5: "
            f"{relevant_count}/{result_count} "
            f"= {precision:.2f}"
        )

    write_results(all_records)

    write_summary(
        question_summaries=question_summaries,
        total_relevant=total_relevant,
        total_results=total_results,
    )

    print()
    print("=" * 75)
    print("Evaluation completed.")
    print(f"Detailed results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("=" * 75)


if __name__ == "__main__":
 main()