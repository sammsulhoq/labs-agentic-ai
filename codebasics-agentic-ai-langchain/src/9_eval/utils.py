"""Utilities for comparing the semantic similarity of two texts."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model on first use and cache it."""
    # Importing this module does not immediately download or load the model.
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def cosine_similarity(
    reference_text: str,
    generated_text: str,
) -> float:
    """Return cosine similarity between two texts, from -1 to 1."""
    if not reference_text.strip() or not generated_text.strip():
        raise ValueError("Both texts must contain non-whitespace characters.")

    embedding_model = get_embedding_model()

    # Unit-length embeddings let us calculate cosine similarity
    # directly using their dot product.
    embeddings = embedding_model.encode(
        [reference_text, generated_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    reference_embedding, generated_embedding = embeddings

    similarity_score = np.dot(
        reference_embedding,
        generated_embedding,
    )

    # Guard against small floating-point deviations outside [-1, 1].
    return float(np.clip(similarity_score, -1.0, 1.0))


def main() -> None:
    """Demonstrate similarity scores for a few sentence pairs."""
    sentence_pairs = [
        (
            "The cat sat on the mat.",
            "A feline rested on the rug.",
        ),
        (
            "The cat sat on the mat.",
            "The dog barked at the mailman.",
        ),
        (
            "Python is great.",
            "I love programming in Python.",
        ),
    ]

    for reference_text, generated_text in sentence_pairs:
        similarity_score = cosine_similarity(
            reference_text,
            generated_text,
        )

        print(
            f'{similarity_score:.4f} | '
            f'"{reference_text}" vs "{generated_text}"'
        )


if __name__ == "__main__":
    main()
