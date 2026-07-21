"""Create embeddings and store them in a FAISS vector index."""

from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import create_all_chunks
from rag.loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIRECTORY = PROJECT_ROOT / "data" / "processed_text"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
VECTOR_DIRECTORY = PROJECT_ROOT / "vector_store"

INDEX_PATH = VECTOR_DIRECTORY / "academic_stress.index"
CHUNKS_PATH = VECTOR_DIRECTORY / "chunks.json"
CONFIG_PATH = VECTOR_DIRECTORY / "config.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40


def main() -> None:
    """Build and save the complete vector database."""

    print("Step 1: Loading documents...")

    documents = load_documents(
        text_directory=TEXT_DIRECTORY,
        metadata_path=METADATA_PATH,
    )

    if not documents:
        raise RuntimeError(
            "No documents were loaded. "
            "Check data/processed_text and metadata.csv."
        )

    print(f"Loaded {len(documents)} documents.")

    print("Step 2: Creating chunks...")

    chunks = create_all_chunks(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise RuntimeError("No chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    print("Step 3: Loading embedding model...")
    print("The first run may take several minutes.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    chunk_texts = [chunk["text"] for chunk in chunks]

    print("Step 4: Creating embeddings...")

    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Normalisation allows inner product to represent cosine similarity.
    faiss.normalize_L2(embeddings)

    print("Step 5: Creating FAISS index...")

    vector_dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    VECTOR_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    CHUNKS_PATH.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "vector_dimension": vector_dimension,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("FAISS vector database created successfully.")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector dimensions: {vector_dimension}")
    print(f"Index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()