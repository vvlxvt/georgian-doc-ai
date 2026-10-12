from app.answer import AnswerGenerator
from app.context import ContextBuilder
from app.retrieval import Retriever


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        answer_generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.answer_generator = answer_generator

    def answer(
        self,
        question: str,
        limit: int = 5,
        score_threshold: float | None = None,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> str:
        results = self.retriever.search(
            query=question,
            limit=limit,
            score_threshold=score_threshold,
            document_id=document_id,
            document_ids=document_ids,
        )

        context = self.context_builder.build(results)

        return self.answer_generator.generate(
            question=question,
            context=context,
        )
