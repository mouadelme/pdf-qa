import fitz
from chunker import chunk_text


def read_pdf(pdf_path: str) -> str:
    """Read all text from a PDF."""

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":
    text = read_pdf("data/sample.pdf")

    chunks = chunk_text(text)

    print(f"Chunks: {len(chunks)}")

    print()

    print(chunks[0])