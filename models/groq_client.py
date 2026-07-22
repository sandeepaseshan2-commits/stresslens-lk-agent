"""Shared Groq model client used by all AI agents."""

import os
from dataclasses import dataclass
from time import perf_counter

from dotenv import load_dotenv
from groq import Groq


# Read variables from the local .env file.
load_dotenv()


FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL",
    "llama-3.1-8b-instant",
)

STRONG_MODEL = os.getenv(
    "GROQ_STRONG_MODEL",
    "llama-3.3-70b-versatile",
)


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    """Information returned after one model call."""

    text: str
    requested_model: str
    returned_model: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GroqModelClient:
    """Send chat requests to different Groq models."""

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Add it to your local .env file."
            )

        self.client = Groq(api_key=api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.1,
    ) -> ModelCallResult:
        """Send messages to one selected model."""

        start_time = perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

        except Exception as error:
            raise RuntimeError(
                f"Groq model call failed for {model}: {error}"
            ) from error

        latency = perf_counter() - start_time

        answer = response.choices[0].message.content or ""

        usage = getattr(response, "usage", None)

        prompt_tokens = int(
            getattr(usage, "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(
            getattr(usage, "total_tokens", 0) or 0
        )

        returned_model = str(
            getattr(response, "model", model)
        )

        return ModelCallResult(
            text=answer.strip(),
            requested_model=model,
            returned_model=returned_model,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )