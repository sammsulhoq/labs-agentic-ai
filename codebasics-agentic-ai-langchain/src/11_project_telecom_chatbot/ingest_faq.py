"""Load FAQ CSV entries into the FAQ Chroma collection."""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from retriever import DATA_DIR, FAQ_COLLECTION, sync_documents


FAQ_CSV_PATH = DATA_DIR / "faq.csv"
REQUIRED_COLUMNS = {"id", "question", "answer", "category"}


def load_faq_documents(csv_path: str | Path) -> list[Document]:
    """Convert each FAQ row into one searchable document."""
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"FAQ CSV not found: {csv_path}")

    # Preserve identifiers as strings, including any leading zeros.
    faq_table = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = REQUIRED_COLUMNS - set(faq_table.columns)

    if missing_columns:
        raise ValueError(
            f"FAQ CSV is missing columns: {sorted(missing_columns)}"
        )

    documents = []

    for row_number, faq_row in enumerate(
        faq_table.to_dict(orient="records"),
        start=2,
    ):
        faq_id = faq_row["id"].strip()
        question = faq_row["question"].strip()
        answer = faq_row["answer"].strip()
        category = faq_row["category"].strip() or "uncategorized"

        if not faq_id or not question or not answer:
            raise ValueError(
                f"FAQ CSV row {row_number} requires an ID, "
                "question, and answer."
            )

        documents.append(
            Document(
                page_content=f"Q: {question}\nA: {answer}",
                metadata={
                    "source": "faq",
                    "category": category,
                    "faq_id": faq_id,
                },
            )
        )

    return documents


def main() -> None:
    """Load, embed, and synchronize the FAQ collection."""
    print("Loading FAQ documents...")
    documents = load_faq_documents(FAQ_CSV_PATH)

    document_ids = [
        f"faq:{document.metadata['faq_id']}"
        for document in documents
    ]

    print(f"Loaded {len(documents)} FAQ entries.")
    print("Embedding and synchronizing the FAQ collection...")

    stored_count = sync_documents(
        collection_name=FAQ_COLLECTION,
        documents=documents,
        document_ids=document_ids,
    )

    print(f"Done. {stored_count} FAQ vectors stored.")


if __name__ == "__main__":
    main()
