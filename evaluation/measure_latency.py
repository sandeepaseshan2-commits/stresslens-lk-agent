"""Measure simple client-side latency for both selected models."""

import csv
from pathlib import Path
from statistics import mean

from models.groq_client import (
    FAST_MODEL,
    STRONG_MODEL,
    GroqModelClient,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "model_latency.csv"
)

NUMBER_OF_RUNS = 3


def main() -> None:
    """Test each model several times and save the results."""

    client = GroqModelClient()

    selected_models = [
        ("Fast routing model", FAST_MODEL),
        ("Strong synthesis model", STRONG_MODEL),
    ]

    records: list[dict[str, object]] = []

    for task_name, model in selected_models:
        model_times: list[float] = []

        print()
        print(f"Testing {task_name}: {model}")

        for run_number in range(
            1,
            NUMBER_OF_RUNS + 1
        ):
            result = client.generate(
                model=model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using one short sentence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Name one factor that may contribute "
                            "to academic stress."
                        ),
                    },
                ],
            )

            model_times.append(
                result.latency_seconds
            )

            records.append(
                {
                    "task_name": task_name,
                    "model": model,
                    "run": run_number,
                    "latency_seconds": round(
                        result.latency_seconds,
                        4,
                    ),
                    "prompt_tokens": (
                        result.prompt_tokens
                    ),
                    "completion_tokens": (
                        result.completion_tokens
                    ),
                    "total_tokens": (
                        result.total_tokens
                    ),
                }
            )

            print(
                f"Run {run_number}: "
                f"{result.latency_seconds:.3f} seconds"
            )

        print(
            "Average latency: "
            f"{mean(model_times):.3f} seconds"
        )

    with OUTPUT_FILE.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "task_name",
            "model",
            "run",
            "latency_seconds",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()