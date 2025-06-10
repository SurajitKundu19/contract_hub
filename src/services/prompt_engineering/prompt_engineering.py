from abc import ABC, abstractmethod
class BasePromptEngineering:
    @abstractmethod
    def generate_prompt(self, query:str, relevant_texts: list[str]):
        pass
class MistralPromptEngineering:
    def __init__(self):
        pass

    def generate_prompt(self, query: str, relevant_texts: list[str]):
        context = "\n\n".join(relevant_texts)
        prompt = f"[INST] Context:\n{context}\n\nQuestion: {query}\n\nAnswer: [/INST]"
        return prompt


class OllamaPromptEngineering:
    def __init__(self):
        pass

    def generate_prompt(self, query: str, relevant_texts: list[str]):
        context = "\n\n".join(relevant_texts)
        prompt = f"Using this data: {context}. Respond to this prompt: {query}"
        return prompt


class LlamaPromptEngineering:
    def __init__(self):
        pass
    def generate_prompt(self, query: str, relevant_texts: list[str]):
        context = "\n\n".join(relevant_texts)
        prompt = f"[INST] Context:\n{context}\n\nQuestion: {query}\n\nAnswer: [/INST]"
        return prompt


class PromptEngineer:

    def __init__(self, prompt_model):
        self.prompt_model = prompt_model

    def get_prompt(self, query, relevant_texts):
        prompt = self.prompt_model.generate_prompt(query, relevant_texts)
        return prompt
