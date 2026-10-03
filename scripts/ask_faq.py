import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "faq_embeddings.json"

load_dotenv(PROJECT_ROOT / ".env")

# Same MongoDB Atlas Voyage endpoint used in embed_faq.py.
EMBEDDINGS_URL = "https://ai.mongodb.com/v1/embeddings"
EMBEDDING_MODEL = "voyage-4-large"
TOP_K = 3

client = Anthropic()


def embed_query(text: str) -> list[float]:
    response = requests.post(
        EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "input": [text],
            "model": EMBEDDING_MODEL,
            "input_type": "query",
        },
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def top_matches(query_embedding: list[float], records: list[dict], k: int) -> list[dict]:
    query_vec = np.array(query_embedding)
    record_vecs = np.array([record["embedding"] for record in records])
    scores = cosine_similarity(query_vec, record_vecs)
    ranked_indices = np.argsort(scores)[::-1][:k]
    return [records[i] for i in ranked_indices]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"## {chunk['heading']}\n{chunk['text']}" for chunk in chunks)
    return (
        "Answer the user's question using ONLY the context below. "
        "If the context doesn't contain the answer, say you don't know. "
        "Answer in the same language as the question, regardless of what "
        "language the context is written in.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    records = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} chunk embeddings from {EMBEDDINGS_PATH.name}")

    while True:
        question = input("\nAsk a question (or 'quit'): ").strip()
        if not question or question.lower() in {"quit", "exit"}:
            break

        query_embedding = embed_query(question)
        chunks = top_matches(query_embedding, records, TOP_K)

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        )

        for block in response.content:
            if block.type == "text":
                print(f"\n{block.text}")


if __name__ == "__main__":
    main()
