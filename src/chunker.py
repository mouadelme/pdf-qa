import re


def split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> list[str]:
    """Split an oversized paragraph at sentence or word boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # The sentence itself may be longer than chunk_size.
        words = sentence.split()
        word_buffer: list[str] = []
        word_buffer_length = 0

        for word in words:
            added_length = len(word) + (1 if word_buffer else 0)

            if word_buffer and word_buffer_length + added_length > chunk_size:
                chunks.append(" ".join(word_buffer))
                word_buffer = [word]
                word_buffer_length = len(word)
            else:
                word_buffer.append(word)
                word_buffer_length += added_length

        if word_buffer:
            current = " ".join(word_buffer)

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap_paragraphs: int = 1,
) -> list[str]:
    """
    Split text into coherent chunks using paragraph boundaries.

    Overlap is measured in paragraphs rather than characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    expanded_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            expanded_paragraphs.append(paragraph)
        else:
            expanded_paragraphs.extend(
                split_long_paragraph(paragraph, chunk_size)
            )

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_length = 0

    for paragraph in expanded_paragraphs:
        separator_length = 2 if current_paragraphs else 0
        candidate_length = current_length + separator_length + len(paragraph)

        if current_paragraphs and candidate_length > chunk_size:
            chunks.append("\n\n".join(current_paragraphs))

            overlap = (
                current_paragraphs[-overlap_paragraphs:]
                if overlap_paragraphs > 0
                else []
            )

            current_paragraphs = overlap.copy()
            current_length = len("\n\n".join(current_paragraphs))

        current_paragraphs.append(paragraph)
        current_length = len("\n\n".join(current_paragraphs))

    if current_paragraphs:
        final_chunk = "\n\n".join(current_paragraphs)

        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return chunks