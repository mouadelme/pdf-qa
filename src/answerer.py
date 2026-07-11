from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-small"


class Answerer:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def answer(
        self,
        question: str,
        contexts: list[str],
    ) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not contexts:
            raise ValueError("At least one context chunk is required")

        # Keep the generator input small and focused.
        context = "\n\n".join(
            passage[:650]
            for passage in contexts[:2]
        )

        prompt = (
            "Answer the question using only the supplied context. "
            "Give one concise sentence. "
            "Ignore unrelated details. "
            "If the context does not contain the answer, say "
            "\"I could not find that information in the document.\"\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=2,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        if not answer:
            return "I could not generate an answer from the retrieved context."

        return answer