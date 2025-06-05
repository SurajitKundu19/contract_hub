from src.services.embeddings.embeddings import Embedder,LegalBertEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilaritySearch:
    def __init__(self, embedder: Embedder, all_document: list[str], all_document_embeddings: list[np.float32]):
        self.embedder = embedder
        self.all_document = all_document
        self.all_document_embeddings = all_document_embeddings

    def _get_embeddings(self, text: list[str]):
        embeddings = self.embedder.get_embedding(text)
        return embeddings

    def find_similarity(self, input_text):
        embeddings = self._get_embeddings(input_text)
        print(len(embeddings))
        print(len(self.all_document_embeddings))
        similarities = cosine_similarity(embeddings, self.all_document_embeddings)[0]
        print(len(similarities))
        best_idx = np.argmax(similarities)
        return self.all_document[best_idx], similarities[best_idx]


if __name__ == "__main__":
    from src.services.data_extraction.extract_data import TextFileSource
    from src.services.chunking.chunker import FixedSizeChunking, Chunker

    data = TextFileSource().get_data()

    print(f" Length of the data {len(data)}")
    chunker_strategy = FixedSizeChunking(chunk_size=400)
    chunker = Chunker(chunker_strategy)
    chunk_data = chunker.chunk_texts(data[:1])
    legal_embedder = LegalBertEmbedder()
    embedder = Embedder(legal_embedder)
    _embeddings = embedder.get_embedding(chunk_data)
    print(f"Embeddings {_embeddings[1].shape}")

    similarity = SimilaritySearch(embedder, chunk_data, _embeddings)
    query = ["What is pursuant to the terms of this Agreement"]
    act_text, embd = similarity.find_similarity(query)
    print(act_text)

