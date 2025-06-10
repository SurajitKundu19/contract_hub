from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import ollama

class BaseAnswerGenerator:

    @abstractmethod
    def generator_answer(self, prompt: str):
        pass


class MistralAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model):
        hf_token = "xxxxxxxxxxx"
        llm_tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
        llm_model_obj = AutoModelForCausalLM.from_pretrained(model, token=hf_token)

        # Create pipeline WITHOUT token
        self.model = pipeline(
            "text-generation",
            model=llm_model_obj,
            tokenizer=llm_tokenizer,
            device="cpu"
        )

    def generator_answer(self, prompt: str):
        self.model(prompt, max_new_tokens=256, do_sample=True, temperature=0.2)


class OllamaAnswerGenerator(BaseAnswerGenerator):
    def __init__(self, model: str):
        self.model = model

    def generator_answer(self, prompt: str):
        answer = ollama.generate(model=self.model, prompt=prompt)
        return answer.response


class AnswerGenerator:

    def __init__(self, generator_model: BaseAnswerGenerator):
        self.generator_model = generator_model

    def get_answer(self, prompt):
        answer = self.generator_model.generator_answer(prompt)
        return answer
