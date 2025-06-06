from abc import ABC, abstractmethod

import numpy as np
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModel
import torch

class BaseEmbedding(ABC):

    @abstractmethod
    def generate_embedding(self, text: str):
        pass


class LegalBertEmbedder(BaseEmbedding):
    def __init__(self):
        super().__init__()
        self.client = InferenceClient(
            provider="hf-inference",
            api_key="hf_xxxxxxxxxxxxxxxxxxxxxxxx",
        )

    def generate_embedding(self, text):
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        # result = self.client.fill_mask(
        #     text,
        #     model=model,
        # )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLS embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
        return mean_embedding


class Embedder:
    def __init__(self, strategy: BaseEmbedding):
        self.strategy = strategy

    def get_embedding(self, texts: list[str]) -> list[np.float32]:
        all_embeddings = []
        for text in texts:
            embeddings = self.strategy.generate_embedding(text)
            all_embeddings.append(embeddings)
        return all_embeddings


if __name__ == "__main__":
    from src.services.data_extraction.extract_data import TextFileSource
    from src.services.chunking.chunker import FixedSizeChunking, Chunker

    data = TextFileSource().get_data()

    print(f" Length of the data {len(data)}")
    chunker_strategy = FixedSizeChunking(chunk_size=400)
    chunker = Chunker(chunker_strategy)
    chunk_data = chunker.chunk_texts(data[:2])
    # print(chunk_data)
    legal_embedder = LegalBertEmbedder()
    embedder = Embedder(legal_embedder)
    _embeddings = embedder.get_embedding(chunk_data)
    # print(f"Embeddings {_embeddings}")

    print(len(chunk_data))

