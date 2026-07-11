import fitz

from chunker import chunk_text
from embedder import Embedder
from retriever import retrieve_chunks


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
    document_embeddings = embedder.embed_documents(chunks)

    question = input("\nAsk a question about the PDF: ")
    query_embedding = embedder.embed_query(question)

    results = retrieve_chunks(
        query_embedding=query_embedding,
        document_embeddings=document_embeddings,
        chunks=chunks,
        top_k=3,
    )

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\n--- Result {rank} | Score: {score:.4f} ---")
        print(chunk)