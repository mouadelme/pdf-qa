import numpy as np
from numpy.typing import NDArray


def retrieve_chunks(
    query_embedding: NDArray[np.float32],
    document_embeddings: NDArray[np.float32],
    chunks: list[str],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the chunks most similar to the query."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if len(chunks) != len(document_embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings"
        )

    # Embeddings were normalized, so the dot product is cosine similarity.
    scores = document_embeddings @ query_embedding[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (chunks[index], float(scores[index]))
        for index in top_indices
    ]