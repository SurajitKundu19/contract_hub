from abc import ABC, abstractmethod
import re


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass


class FixedSizeChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, text: list[str]) -> list[str]:
        words = text.split()
        return [' '.join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]


class SentenceChunking(ChunkingStrategy):
    def chunk(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return sentences


class OverlappingChunking(ChunkingStrategy):
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+self.chunk_size]
            chunks.append(' '.join(chunk))
            i += self.chunk_size - self.overlap
        return chunks


class Chunker:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy

    def chunk_texts(self, texts: list[str]) -> list[str]:
        all_chunks = []
        for text in texts:
            chunks = self.strategy.chunk(text)
            all_chunks.extend(chunks)
        return all_chunks


if __name__ == "__main__":
    from src.services.data_extraction.extract_data import TextFileSource

    data = TextFileSource().get_data()
    print(f" Length of the data {len(data)}")
    chunker_strategy = FixedSizeChunking(chunk_size=400)
    chunker = Chunker(chunker_strategy)
    chunk_data = chunker.chunk_texts(data)
    print(len(chunk_data))
