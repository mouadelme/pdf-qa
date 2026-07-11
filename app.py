import html
import sys
from pathlib import Path
from typing import Any

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from rag_service import RAGService  # noqa: E402


MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024

# The models are loaded once when the application starts.
rag_service = RAGService()


def process_document(
    pdf_path: str | None,
) -> tuple[dict[str, Any] | None, str, str]:
    """Process an uploaded PDF and store its index in session state."""
    if not pdf_path:
        return None, "### Upload a PDF to begin.", ""

    try:
        file_path = Path(pdf_path)

        if file_path.stat().st_size > MAX_PDF_SIZE_BYTES:
            raise ValueError("The PDF must be 25 MB or smaller.")

        document_state = rag_service.process_pdf(pdf_path)

        chunk_count = len(document_state["chunks"])
        filename = html.escape(document_state["filename"])

        status = (
            f"### Document ready\n"
            f"**File:** {filename}  \n"
            f"**Searchable chunks:** {chunk_count}"
        )

        return document_state, status, ""

    except Exception as error:
        return None, f"### Could not process PDF\n{html.escape(str(error))}", ""


def answer_question(
    question: str,
    document_state: dict[str, Any] | None,
) -> tuple[str, str]:
    """Answer a question and format the retrieved source passage."""
    try:
        answer, results = rag_service.ask(
            question=question,
            document_state=document_state or {},
            top_k=1,
        )

        source_sections: list[str] = []

        for index, (passage, score) in enumerate(results, start=1):
            escaped_passage = html.escape(passage)

            source_sections.append(
                f"### Source {index}\n"
                f"**Similarity:** `{score:.4f}`\n\n"
                f"> {escaped_passage.replace(chr(10), chr(10) + '> ')}"
            )

        sources = "\n\n".join(source_sections)

        return answer, sources

    except Exception as error:
        return f"Error: {html.escape(str(error))}", ""


def clear_application() -> tuple[
    None,
    None,
    str,
    str,
    str,
]:
    """Clear the current document and conversation."""
    return (
        None,
        None,
        "### Upload a PDF to begin.",
        "",
        "",
    )


with gr.Blocks(title="PDF Question Answering") as demo:
    document_state = gr.State(value=None)

    gr.Markdown(
        """
        # PDF Question Answering

        Upload a text-based PDF and ask questions about its contents.

        The application uses semantic retrieval and a local language model.
        Answers are generated only from the retrieved document context.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )

            with gr.Row():
                process_button = gr.Button(
                    "Process PDF",
                    variant="primary",
                )
                clear_button = gr.Button("Clear")

            document_status = gr.Markdown(
                "### Upload a PDF to begin."
            )

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Question",
                placeholder="What is the main advantage described in the document?",
                lines=2,
            )

            ask_button = gr.Button(
                "Ask question",
                variant="primary",
            )

            answer_output = gr.Markdown(
                label="Answer",
                value="",
            )

    with gr.Accordion("Retrieved source", open=False):
        sources_output = gr.Markdown()

    process_button.click(
        fn=process_document,
        inputs=[pdf_file],
        outputs=[
            document_state,
            document_status,
            sources_output,
        ],
    )

    ask_button.click(
        fn=answer_question,
        inputs=[
            question_input,
            document_state,
        ],
        outputs=[
            answer_output,
            sources_output,
        ],
    )

    question_input.submit(
        fn=answer_question,
        inputs=[
            question_input,
            document_state,
        ],
        outputs=[
            answer_output,
            sources_output,
        ],
    )

    clear_button.click(
        fn=clear_application,
        outputs=[
            document_state,
            pdf_file,
            document_status,
            answer_output,
            sources_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()