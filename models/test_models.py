"""Test the fast and strong Groq models separately."""

from models.groq_client import (
    FAST_MODEL,
    STRONG_MODEL,
    GroqModelClient,
)


def test_one_model(
    client: GroqModelClient,
    model: str,
    task_name: str,
) -> None:
    """Send one small test request to a model."""

    print()
    print("=" * 65)
    print(f"Testing: {task_name}")
    print(f"Model: {model}")
    print("=" * 65)

    result = client.generate(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are helping with an academic stress "
                    "research support project."
                ),
            },
            {
                "role": "user",
                "content": (
                    "In one short sentence, explain what "
                    "academic stress means."
                ),
            },
        ],
    )

    print(f"Answer: {result.text}")
    print(
        f"Latency: {result.latency_seconds:.3f} seconds"
    )
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(
        f"Completion tokens: "
        f"{result.completion_tokens}"
    )
    print(f"Total tokens: {result.total_tokens}")


def main() -> None:
    """Test the two deliberately selected models."""

    client = GroqModelClient()

    test_one_model(
        client=client,
        model=FAST_MODEL,
        task_name="Fast routing model",
    )

    test_one_model(
        client=client,
        model=STRONG_MODEL,
        task_name="Strong synthesis model",
    )

    print()
    print("Both model tests completed.")


if __name__ == "__main__":
    main()