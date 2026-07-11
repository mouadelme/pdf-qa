from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> NDArray[np.float32]:
        """Convert document chunks into normalized embedding vectors."""
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Convert one search query into a normalized embedding vector."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(np.float32)
