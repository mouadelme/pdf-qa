import fitz

from chunker import chunk_text
from embedder import Embedder


def read_pdf(pdf_path: str) -> str:
    """Extract text from every page of a PDF."""
    text_parts: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    chunks = chunk_text(text)

    embedder = Embedder()
    embeddings = embedder.embed_documents(chunks)

    print(f"Chunks: {len(chunks)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"First embedding dimensions: {embeddings[0][:10]}")