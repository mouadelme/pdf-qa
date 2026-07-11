from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from answerer import Answerer
from chunker import chunk_text
from embedder import Embedder
from pdf_reader import read_pdf
from retriever import retrieve_chunks


class RAGService:
    """Manage PDF processing, retrieval, and answer generation."""

    def __init__(self) -> None:
        # Load each model only once when the application starts.
        self.embedder = Embedder()
        self.answerer = Answerer()

    def process_pdf(self, pdf_path: str) -> dict[str, Any]:
        """Extract, chunk, and embed an uploaded PDF."""
        if not pdf_path:
            raise ValueError("Please upload a PDF.")

        path = Path(pdf_path)

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        text = read_pdf(str(path))

        if not text.strip():
            raise ValueError(
                "No readable text was found. The PDF may contain only "
                "scanned images."
            )

        chunks = chunk_text(text)

        if not chunks:
            raise ValueError("The PDF did not produce any searchable chunks.")

        embeddings = self.embedder.embed_documents(chunks)

        return {
            "filename": path.name,
            "chunks": chunks,
            "embeddings": embeddings,
        }

    def ask(
        self,
        question: str,
        document_state: dict[str, Any],
        top_k: int = 1,
    ) -> tuple[str, list[tuple[str, float]]]:
        """Answer a question using an already processed PDF."""
        if not question.strip():
            raise ValueError("Please enter a question.")

        if not document_state:
            raise ValueError("Please process a PDF before asking questions.")

        chunks: list[str] = document_state["chunks"]
        embeddings: NDArray[np.float32] = document_state["embeddings"]

        query_embedding = self.embedder.embed_query(question)

        results = retrieve_chunks(
            query_embedding=query_embedding,
            document_embeddings=embeddings,
            chunks=chunks,
            top_k=top_k,
        )

        contexts = [chunk for chunk, _score in results]

        answer = self.answerer.answer(
            question=question,
            contexts=contexts,
        )

        return answer, results