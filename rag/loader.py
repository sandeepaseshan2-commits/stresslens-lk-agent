"""Load processed research documents and their metadata."""

from csv import DictReader
from pathlib import Path
from typing import Any


def load_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Load metadata.csv and organise it by text filename."""

    metadata: dict[str, dict[str, str]] = {}

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        reader = DictReader(csv_file)

        for row in reader:
            text_file = row.get("text_file", "").strip()

            if text_file:
                metadata[text_file] = row

    return metadata


def load_documents(
    text_directory: Path,
    metadata_path: Path
) -> list[dict[str, Any]]:
    """Load all valid text documents."""

    metadata = load_metadata(metadata_path)
    documents: list[dict[str, Any]] = []

    text_files = sorted(text_directory.glob("*.txt"))

    for text_path in text_files:
        text = text_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if len(text.split()) < 50:
            print(f"Skipped short document: {text_path.name}")
            continue

        document_metadata = metadata.get(text_path.name, {})

        documents.append(
            {
                "document_id": document_metadata.get(
                    "document_id",
                    text_path.stem
                ),
                "title": document_metadata.get(
                    "title",
                    text_path.stem
                ),
                "authors": document_metadata.get(
                    "authors",
                    "Unknown"
                ),
                "year": document_metadata.get(
                    "year",
                    "Unknown"
                ),
                "topic": document_metadata.get(
                    "topic",
                    "Academic stress"
                ),
                "source": text_path.name,
                "text": text,
            }
        )

    return documents