"""Load resolved support tickets into the tickets Chroma collection."""

import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.documents import Document

from retriever import DATA_DIR, TICKET_COLLECTION, sync_documents


TICKETS_DATABASE_PATH = DATA_DIR / "tickets.db"


def load_ticket_documents(db_path: str | Path) -> list[Document]:
    """Convert resolved tickets into searchable issue-resolution documents."""
    database_path = Path(db_path).resolve()

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Tickets database not found: {database_path}"
        )

    # Read-only mode avoids accidentally creating or modifying the database.
    database_uri = f"{database_path.as_uri()}?mode=ro"

    with closing(
        sqlite3.connect(database_uri, uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row

        ticket_rows = connection.execute(
            """
            SELECT
                ticket_id,
                issue_type,
                description,
                resolution,
                category,
                status
            FROM tickets
            WHERE status = ?
            ORDER BY ticket_id
            """,
            ("resolved",),
        ).fetchall()

    documents = []

    for ticket_row in ticket_rows:
        ticket_id = ticket_row["ticket_id"]
        issue_type = str(ticket_row["issue_type"] or "").strip()
        description = str(ticket_row["description"] or "").strip()
        resolution = str(ticket_row["resolution"] or "").strip()
        category = str(
            ticket_row["category"] or "uncategorized"
        ).strip()

        if ticket_id is None or not description or not resolution:
            raise ValueError(
                f"Resolved ticket {ticket_id!r} requires an ID, "
                "description, and resolution."
            )

        documents.append(
            Document(
                page_content=(
                    f"Issue: {issue_type}\n"
                    f"Description: {description}\n"
                    f"Resolution: {resolution}"
                ),
                metadata={
                    "source": "ticket",
                    "ticket_id": str(ticket_id),
                    "category": category or "uncategorized",
                    "status": ticket_row["status"],
                },
            )
        )

    return documents


def main() -> None:
    """Load, embed, and synchronize resolved support tickets."""
    print("Loading resolved tickets...")
    documents = load_ticket_documents(TICKETS_DATABASE_PATH)

    document_ids = [
        f"ticket:{document.metadata['ticket_id']}"
        for document in documents
    ]

    print(f"Loaded {len(documents)} resolved tickets.")
    print("Embedding and synchronizing the tickets collection...")

    stored_count = sync_documents(
        collection_name=TICKET_COLLECTION,
        documents=documents,
        document_ids=document_ids,
    )

    print(f"Done. {stored_count} ticket vectors stored.")


if __name__ == "__main__":
    main()
