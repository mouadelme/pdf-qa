import fitz


def read_pdf(pdf_path: str) -> str:
    """Read all text from a PDF."""

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":
    pdf_text = read_pdf("data/sample.pdf")

    print(pdf_text[:1000])