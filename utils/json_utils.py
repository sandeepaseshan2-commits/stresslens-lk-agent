"""Safely extract a JSON object from model output."""

import json
import re
from typing import Any


def extract_json_object(
    model_text: str,
) -> dict[str, Any]:
    """Extract and parse the first JSON object."""

    cleaned_text = re.sub(
        r"```(?:json)?",
        "",
        model_text,
        flags=re.IGNORECASE,
    )

    cleaned_text = cleaned_text.replace(
        "```",
        "",
    ).strip()

    start_position = cleaned_text.find("{")
    end_position = cleaned_text.rfind("}")

    if start_position == -1 or end_position == -1:
        raise ValueError(
            "The model response did not contain JSON."
        )

    json_text = cleaned_text[
        start_position:end_position + 1
    ]

    parsed_data = json.loads(json_text)

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "The parsed JSON is not an object."
        )

    return parsed_data