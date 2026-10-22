"""Retrieve product rating summaries from the store database."""

from setup_db import open_database


def get_product_rating(product_id: int) -> dict:
    """Return the average rating and review count for one product."""
    return get_ratings_for_products([product_id])[0]


def get_ratings_for_products(product_ids: list[int]) -> list[dict]:
    """Return rating summaries in the same order as the supplied IDs."""
    if not product_ids:
        return []

    # SQL values are bound separately rather than interpolated.
    placeholders = ", ".join("?" for _ in product_ids)

    sql_query = f"""
        SELECT
            product_id,
            AVG(rating) AS average_rating,
            COUNT(*) AS review_count
        FROM reviews
        WHERE product_id IN ({placeholders})
        GROUP BY product_id
    """

    with open_database() as connection:
        rating_rows = connection.execute(
            sql_query,
            product_ids,
        ).fetchall()

    ratings_by_product_id = {
        row["product_id"]: {
            "average_rating": round(row["average_rating"], 2),
            "review_count": row["review_count"],
        }
        for row in rating_rows
    }

    # Preserve the original convention: no reviews produces a 0.0 rating.
    # Check review_count to distinguish "unrated" from an actual rating.
    return [
        {
            "product_id": product_id,
            "average_rating": ratings_by_product_id.get(
                product_id, {}
            ).get("average_rating", 0.0),
            "review_count": ratings_by_product_id.get(
                product_id, {}
            ).get("review_count", 0),
        }
        for product_id in product_ids
    ]


def main() -> None:
    """Demonstrate single-product and batch rating lookups."""
    product_rating = get_product_rating(1)

    print("Single product rating:")
    print(
        f"Product {product_rating['product_id']}: "
        f"{product_rating['average_rating']} stars "
        f"({product_rating['review_count']} reviews)"
    )

    print("\nBatch ratings:")

    for product_rating in get_ratings_for_products([1, 3, 5, 7]):
        print(
            f"Product {product_rating['product_id']}: "
            f"{product_rating['average_rating']} stars "
            f"({product_rating['review_count']} reviews)"
        )


if __name__ == "__main__":
    main()
