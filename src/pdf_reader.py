import re

import fitz

from answerer import Answerer
from chunker import chunk_text
from embedder import Embedder
from retriever import retrieve_chunks


BULLET_PATTERN = re.compile(r"^[\s\u2022\u25cf\u25a0\u25aa\u25e6\u25cb\u25a1\-–—*]+")


def clean_line(line: str) -> str:
    """Normalize one extracted PDF line."""
    line = line.replace("\u00a0", " ")
    line = line.replace("\u200b", " ")  # zero-width space
    line = line.replace("\u200c", " ")
    line = line.replace("\u200d", " ")
    line = line.replace("\ufeff", " ")

    line = re.sub(
        r"^[\s\u2022\u25cf\u25a0\u25aa\u25e6\u25cb\u25a1\-–—*]+",
        "",
        line,
    )

    line = re.sub(r"\s+", " ", line)
    return line.strip()

def clean_page_text(text: str) -> str:
    """Clean PDF text while preserving paragraph boundaries."""
    raw_lines = text.splitlines()
    cleaned_lines: list[str] = []
    current_paragraph: list[str] = []

    for raw_line in raw_lines:
        line = clean_line(raw_line)

        if not line:
            if current_paragraph:
                cleaned_lines.append(" ".join(current_paragraph))
                current_paragraph = []
            continue

        # Join words split by a PDF line break:
        # "provision-" + "ing" -> "provisioning"
        if current_paragraph and current_paragraph[-1].endswith("-"):
            current_paragraph[-1] = current_paragraph[-1][:-1] + line
        else:
            current_paragraph.append(line)

    if current_paragraph:
        cleaned_lines.append(" ".join(current_paragraph))

    return "\n\n".join(cleaned_lines)


def read_pdf(pdf_path: str) -> str:
    """Extract and clean text from every page of a PDF."""
    pages: list[str] = []

    with fitz.open(pdf_path) as document:
        for page in document:
            raw_text = page.get_text("text", sort=True)
            cleaned_text = clean_page_text(raw_text)

            if cleaned_text:
                pages.append(cleaned_text)

    return "\n\n".join(pages)


if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")
    chunks = chunk_text(text)

    embedder = Embedder()
    document_embeddings = embedder.embed_documents(chunks)

    answerer = Answerer()

    while True:
        question = input(
            "\nAsk a question about the PDF "
            "(or type 'quit' to exit): "
        ).strip()

        if question.lower() in {"quit", "exit"}:
            break

        if not question:
            print("Please enter a question.")
            continue

        query_embedding = embedder.embed_query(question)

        results = retrieve_chunks(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            chunks=chunks,
            top_k=1,
        )

        retrieved_chunks = [
            chunk
            for chunk, _score in results
        ]

        answer = answerer.answer(
            question=question,
            contexts=retrieved_chunks,
        )

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved passages:")
        for rank, (chunk, similarity) in enumerate(results, start=1):
            print(
                f"\n--- Passage {rank} "
                f"| Retrieval score: {similarity:.4f} ---"
            )
            print(chunk)