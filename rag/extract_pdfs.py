"""Extract readable text from PDF research papers."""

from pathlib import Path
import re

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw_pdfs"
TEXT_DIRECTORY = PROJECT_ROOT / "data" / "processed_text"


def clean_text(text: str) -> str:
    """Remove common PDF extraction problems."""

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    # Join words that were divided at the end of a PDF line.
    text = re.sub(r"-\s*\n\s*(?=\w)", "", text)

    # Replace repeated spaces and tabs with one space.
    text = re.sub(r"[ \t]+", " ", text)

    # Keep no more than two consecutive line breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: Path) -> str:
    """Read all available text from one PDF."""

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                f"\n--- Page {page_number} ---\n{page_text.strip()}"
            )

    return clean_text("\n".join(pages))


def main() -> None:
    """Convert every PDF in raw_pdfs into a text file."""

    TEXT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw_pdfs.")
        return

    successful = 0

    for pdf_path in pdf_files:
        try:
            extracted_text = extract_pdf(pdf_path)

            if len(extracted_text.split()) < 50:
                print(
                    f"WARNING: {pdf_path.name} produced very little text."
                )
                continue

            output_path = TEXT_DIRECTORY / f"{pdf_path.stem}.txt"
            output_path.write_text(extracted_text, encoding="utf-8")

            successful += 1
            print(
                f"Converted: {pdf_path.name} "
                f"-> {output_path.name}"
            )

        except Exception as error:
            print(f"ERROR reading {pdf_path.name}: {error}")

    print()
    print(f"Finished. {successful} PDF files converted successfully.")


if __name__ == "__main__":
    main()