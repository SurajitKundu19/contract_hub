import os

from dotenv import find_dotenv, load_dotenv

from src.services.data_extraction.extract_data import TextFileSource
from src.services.chunking.chunker import FixedSizeChunking, Chunker
from src.services.embeddings.embeddings import LegalBertEmbedder, Embedder
from src.services.similarity_search.similarity_search import SimilaritySearch
from src.services.prompt_engineering.prompt_engineering import PromptEngineer, LlamaPromptEngineering, MistralPromptEngineering, OllamaPromptEngineering
from src.services.answer_generation.answer_generation import AnswerGenerator, MistralAnswerGenerator, OllamaAnswerGenerator

load_dotenv(find_dotenv("../.env"))
print(f"Hugging Face Token: {os.getenv('HUGGING_FACE_TOKEN')}")

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
query = "What is Company's costs of manufacturing"
similar_text, embd = similarity.find_similarity(query)
print(similar_text)


# prompt_template = MistralPromptEngineering()
# prompt_engineer = PromptEngineer(prompt_template)
# prompt = prompt_engineer.get_prompt(query, similar_text)

# llm_model = MistralAnswerGenerator(model="mistralai/Mistral-7B-Instruct-v0.2")
# answer_generator = AnswerGenerator(llm_model)
# answer_generator.get_answer(prompt)

prompt_template = OllamaPromptEngineering()
prompt_engineer = PromptEngineer(prompt_template)
prompt = prompt_engineer.get_prompt(query, similar_text)

llm_model = OllamaAnswerGenerator(model="llama3.2")
answer_generator = AnswerGenerator(llm_model)
answer = answer_generator.get_answer(prompt)
print(answer)
