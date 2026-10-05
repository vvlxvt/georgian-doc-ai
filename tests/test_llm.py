from app.llm import OllamaLLMService


def test_ollama_llm_service_configuration():
    llm = OllamaLLMService(
        model="llama3.2:3b",
        base_url="http://localhost:11434/",
    )

    assert llm.model == "llama3.2:3b"
    assert llm.base_url == "http://localhost:11434"



llm = OllamaLLMService()

answer = llm.generate(
    "Explain in one sentence what a vector database is."
)

print(answer)