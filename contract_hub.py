import os
from typing import List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from huggingface_hub import login

# Use environment variable for the token
login(token=os.getenv("HF_TOKEN"))


def paragraph_chunking(text: str, max_tokens: int = 512, tokenizer=None) -> List[str]:
    """
    Splits text into chunks based on paragraphs, ensuring each chunk is within
    max_tokens. If a paragraph is too long, it is further split by sentences.
    """
    assert tokenizer is not None, "Tokenizer must be provided"
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        tokens = tokenizer.encode(paragraph, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            chunks.append(paragraph)
        else:
            # If paragraph is too long, split by sentences
            sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
            current_chunk = ""
            for sentence in sentences:
                if not sentence.endswith("."):
                    sentence += "."
                test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                if len(tokenizer.encode(test_chunk, add_special_tokens=False)) > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk = test_chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks


def mean_pooling(model_output, attention_mask):
    """
    Applies mean pooling to the output of a transformer model.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embeddings(
    texts: List[str], tokenizer, model, batch_size: int = 16, device: str = "cpu"
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using Legal-BERT with mean pooling.
    """
    all_embeddings = []
    model = model.to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pooling(output, encoded["attention_mask"])
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


def load_and_chunk_documents(
    directory: str, tokenizer, max_tokens: int = 512
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Reads all text files in a directory, splits them into paragraph-based chunks,
    and returns the chunks and their source mapping.
    Returns:
        chunks: List of text chunks.
        mapping: List of (filename, chunk_idx) tuples for each chunk.
    """
    chunks = []
    mapping = []
    for filename in tqdm(os.listdir(directory), desc="Reading files"):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                text = f.read()
            file_chunks = paragraph_chunking(text, max_tokens=max_tokens, tokenizer=tokenizer)
            chunks.extend(file_chunks)
            mapping.extend([(filename, idx) for idx in range(len(file_chunks))])
    return chunks, mapping


def save_embeddings_and_chunks(
    data_dir: str,
    embedding_model: str = "nlpaueb/legal-bert-base-uncased",
    chunk_tokens: int = 512,
    device: str = "cpu",
    save_path: str = "embeddings_and_chunks.pkl",
):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    chunks, mapping = load_and_chunk_documents(data_dir, tokenizer, chunk_tokens)
    chunk_embeddings = get_embeddings(chunks, tokenizer, model, device=device)
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": chunk_embeddings, "chunks": chunks, "mapping": mapping}, f)
    print(f"Saved embeddings and chunks to {save_path}")


def search_similar_chunks(
    query: str,
    chunk_embeddings: np.ndarray,
    chunks: List[str],
    tokenizer,
    model,
    top_k: int = 5,
    device: str = "cpu",
) -> List[Tuple[str, float, int]]:
    model = model.to(device)
    encoded = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
        query_embedding = mean_pooling(output, encoded["attention_mask"]).cpu().numpy()
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(chunks[i], similarities[i], i) for i in top_indices]


if __name__ == "__main__":
    # Example usage
    data_dir = "data/CUAD_v1/full_contract_txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_embeddings_and_chunks(data_dir=data_dir, device=device)
