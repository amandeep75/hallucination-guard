from hallucination_guard import hallucination_guard
import ollama


# Example: function that calls an LLM via Ollama
def call_llm(prompt: str, temperature: float = 0.7) -> str:
    response = ollama.chat(
        model="llama3",   # change this to your installed model (e.g. "mistral", "llama2", "phi3", etc.)
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


@hallucination_guard(n_samples=5, threshold=0.45, action="tag")
def answer_fact(question: str, temperature: float = 0.7):
    return {"answer": call_llm(question, temperature=temperature)}


if __name__ == "__main__":
    resp = answer_fact("What's the capital of Australia?")
    print(resp)
    # -> {
    #   "answer": "...",
    #   "hallucination_warning": True/False,
    #   "hallucination_meta": { ... scores ... }
    # }