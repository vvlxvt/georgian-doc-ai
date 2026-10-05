from app.answer import AnswerGenerator
from app.llm import LLMService


class FakeLLMService(LLMService):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer"


def test_answer_generator_uses_llm():
    llm = FakeLLMService()
    generator = AnswerGenerator(llm)

    answer = generator.generate(
        question="What does the document say?",
        context="The document says that the contract is valid.",
    )

    assert answer == "Generated answer"
    assert len(llm.prompts) == 1


def test_answer_generator_includes_question_and_context():
    llm = FakeLLMService()
    generator = AnswerGenerator(llm)

    generator.generate(
        question="What does the document say?",
        context="The contract is valid.",
    )

    prompt = llm.prompts[0]

    assert "What does the document say?" in prompt
    assert "The contract is valid." in prompt


def test_answer_generator_does_not_call_llm_without_context():
    llm = FakeLLMService()
    generator = AnswerGenerator(llm)

    answer = generator.generate(
        question="What does the document say?",
        context="",
    )

    assert answer == "I don't have enough information to answer this question."
    assert llm.prompts == []


def test_answer_generator_rejects_empty_question():
    llm = FakeLLMService()
    generator = AnswerGenerator(llm)

    try:
        generator.generate(
            question="   ",
            context="Some context",
        )
    except ValueError as exc:
        assert str(exc) == "question must not be empty"
    else:
        raise AssertionError("ValueError was not raised")