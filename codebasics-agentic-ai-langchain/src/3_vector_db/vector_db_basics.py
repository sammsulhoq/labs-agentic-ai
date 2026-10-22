"""Demonstrate semantic search with ChromaDB."""

import chromadb
from chromadb.api.models.Collection import Collection

COLLECTION_NAME = "news_articles"
RESULTS_PER_QUERY = 2
SHOW_EMBEDDINGS = False

# Sample documents for demonstrating semantic search.
ARTICLE_IDS = ["id1", "id2", "id3", "id4"]

ARTICLES = [
    "Apple is leading in a smart phone game with iPhone sales up by 35%",
    "Tesla booked a minor profit of 1 billion $ in Q2",
    "Apples are high in fiber, vitamin C, and various antioxidants",
    "SpaceX got NASA contract worth 10 billion $",
]

SEARCH_QUERIES = [
    "This is a query related to Elon Musk",
    "I love iPad",
    "Tips for improving health",
]


def load_articles(collection: Collection) -> None:
    """Store the sample articles and generate their embeddings."""
    # Chroma generates embeddings using the collection's embedding function.
    # Upsert inserts new IDs and updates existing IDs when rerun.
    collection.upsert(
        ids=ARTICLE_IDS,
        documents=ARTICLES,
    )


def display_embeddings(collection: Collection) -> None:
    """Display a short preview of each stored embedding."""
    # get() retrieves stored records without similarity ranking.
    data = collection.get(
        include=["documents", "embeddings", "metadatas"],
    )

    documents = data["documents"]
    embeddings = data["embeddings"]

    if documents is None or embeddings is None:
        print("No documents or embeddings available.")
        return

    print("\n=== STORED EMBEDDINGS ===")

    for article_id, document, embedding in zip(
        data["ids"],
        documents,
        embeddings,
    ):
        print(f"\nDocument ID: {article_id}")
        print(f"Text: {document}")
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 10 values: {embedding[:10].tolist()}")


def search_articles(
    collection: Collection,
    query: str,
    n_results: int = RESULTS_PER_QUERY,
) -> None:
    """Find and display articles closest to the query's embedding."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances"],
    )

    print(f"\n=== QUERY: {query} ===")

    # Query results are grouped by input query.
    # Since we submitted one query, its matches are at index 0.
    documents = results["documents"]
    distances = results["distances"]

    if documents is None or distances is None:
        print("No results available.")
        return

    for rank, (article_id, document, distance) in enumerate(
        zip(
            results["ids"][0],
            documents[0],
            distances[0],
        ),
        start=1,
    ):
        print(f"\n{rank}. {document}")
        print(f"   Document ID: {article_id}")
        print(f"   Distance: {distance:.4f}")


def main() -> None:
    """Initialize Chroma, load articles, and run example searches."""
    # Use an in-memory database for this demonstration.
    client = chromadb.Client()

    # Return the existing collection or create it if needed.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
    )

    load_articles(collection)

    # Enable this flag to inspect the generated numerical vectors.
    if SHOW_EMBEDDINGS:
        display_embeddings(collection)

    for query in SEARCH_QUERIES:
        search_articles(collection, query)


if __name__ == "__main__":
    main()