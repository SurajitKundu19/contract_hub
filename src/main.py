from src.services.data_extraction.extract_data import TextFileSource
from src.services.chunking.chunker import FixedSizeChunking, Chunker
from src.services.embeddings.embeddings import LegalBertEmbedder, Embedder
from src.services.similarity_search.similarity_search import SimilaritySearch


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
query = "Hello"
act_text, embd = similarity.find_similarity(query)
print(act_text)