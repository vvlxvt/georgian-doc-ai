from app.llm import LLMService


class AnswerGenerator:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        question = question.strip()

        if not question:
            raise ValueError("question must not be empty")

        if not context.strip():
            return "I don't have enough information to answer this question."

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        return self.llm.generate(prompt)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
    ) -> str:
        return f"""Answer the question using only the provided context.

If the context does not contain enough information to answer the question,
say that there is not enough information.

Context:
{context}

Question:
{question}

Answer:"""