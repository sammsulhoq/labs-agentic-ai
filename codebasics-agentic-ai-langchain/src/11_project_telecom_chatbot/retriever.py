"""Shared vector-store configuration and multi-source retrieval."""

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FAQ_COLLECTION = "faq"
TICKET_COLLECTION = "tickets"
GUIDE_COLLECTION = "guides"

INDEX_BATCH_SIZE = 100


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per Python process."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )


def open_vector_store(
    collection_name: str,
    create_if_missing: bool = False,
) -> Chroma:
    """Open a collection using the project's shared embedding settings."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        create_collection_if_not_exists=create_if_missing,
    )


def sync_documents(
    collection_name: str,
    documents: list[Document],
    document_ids: list[str],
) -> int:
    """Synchronize a collection with its current source documents.

    Adds or updates stable IDs, then removes records no longer present
    in the source. Each collection belongs to one ingestion script.
    """
    if not documents:
        raise ValueError(
            f"No documents loaded for '{collection_name}'. "
            "The existing collection has not been changed."
        )

    if len(documents) != len(document_ids):
        raise ValueError("Each document must have one corresponding ID.")

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique within a collection.")

    vector_store = open_vector_store(
        collection_name,
        create_if_missing=True,
    )

    existing_ids = set(vector_store.get(include=[])["ids"])

    # Reusing IDs updates existing records instead of creating duplicates.
    for start_index in range(0, len(documents), INDEX_BATCH_SIZE):
        end_index = start_index + INDEX_BATCH_SIZE

        vector_store.add_documents(
            documents=documents[start_index:end_index],
            ids=document_ids[start_index:end_index],
        )

    # Remove stale records only after all current documents are stored.
    stale_ids = sorted(existing_ids - set(document_ids))

    for start_index in range(0, len(stale_ids), INDEX_BATCH_SIZE):
        vector_store.delete(
            ids=stale_ids[start_index:start_index + INDEX_BATCH_SIZE]
        )

    return len(vector_store.get(include=[])["ids"])


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    """Retrieve and combine matches from all three collections."""
    collection_limits = {
        FAQ_COLLECTION: k_faq,
        TICKET_COLLECTION: k_tickets,
        GUIDE_COLLECTION: k_guides,
    }

    if any(limit < 1 for limit in collection_limits.values()):
        raise ValueError("Each retrieval count must be at least 1.")

    if not CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            "Vector store not found. Run all three ingestion scripts first."
        )

    source_retrievers = []

    for collection_name, result_limit in collection_limits.items():
        vector_store = open_vector_store(collection_name)

        # Check for missing ingestion without accessing private attributes.
        if not vector_store.get(limit=1, include=[])["ids"]:
            raise ValueError(
                f"Collection '{collection_name}' is empty. "
                "Run its ingestion script first."
            )

        source_retrievers.append(
            vector_store.as_retriever(
                search_kwargs={"k": result_limit},
            )
        )

    def retrieve_documents(question: str) -> list[Document]:
        """Collect matches in FAQ, ticket, and guide order."""
        if not question.strip():
            raise ValueError("The question must not be empty.")

        retrieved_documents = []

        # This merges per-source results; it does not globally rerank them.
        for source_retriever in source_retrievers:
            retrieved_documents.extend(
                source_retriever.invoke(question)
            )

        return retrieved_documents

    return RunnableLambda(retrieve_documents)
