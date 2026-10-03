import json
import os
import re
from pathlib import Path

import requests

# load_dotenv() lets us read secret values (like API keys) from the ".env"
# file at the project root instead of typing them directly into the code.
from dotenv import load_dotenv

# This script lives in scripts/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAQ_PATH = Path(__file__).resolve().parent / "../data/faq.md"
OUTPUT_PATH = PROJECT_ROOT / "data" / "faq_embeddings.json"

load_dotenv(PROJECT_ROOT / ".env")

# The VOYAGE_API_KEY in .env is a MongoDB Atlas "model API key", which only
# authenticates against MongoDB's endpoint, not api.voyageai.com. So we call
# it directly with requests instead of using the voyageai SDK.
EMBEDDINGS_URL = "https://ai.mongodb.com/v1/embeddings"
EMBEDDING_MODEL = "voyage-4-large"

# Matches level-2 markdown headings ("## Heading" or "##Heading") anywhere
# a line starts with them.
HEADING_PATTERN = re.compile(r"(?m)^##\s*(.+?)\s*$")


def split_into_chunks(markdown_text: str) -> list[dict]:
    """Split markdown into chunks, one per '##' heading section.

    Each chunk keeps its heading line plus the body text under it, since
    embedding the heading alongside the body gives the model more context
    about what the chunk is about.
    """
    headings = list(HEADING_PATTERN.finditer(markdown_text))
    chunks = []
    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown_text)
        chunks.append(
            {
                "heading": match.group(1).strip(),
                "text": markdown_text[start:end].strip(),
            }
        )
    return chunks


def main():
    markdown_text = FAQ_PATH.read_text(encoding="utf-8")
    chunks = split_into_chunks(markdown_text)
    print(f"Split {FAQ_PATH.name} into {len(chunks)} chunks.")

    # The endpoint accepts a batch of texts in a single request, which is
    # faster and cheaper than one request per chunk.
    response = requests.post(
        EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "input": [chunk["text"] for chunk in chunks],
            "model": EMBEDDING_MODEL,
            "input_type": "document",
        },
    )
    response.raise_for_status()
    embeddings = [item["embedding"] for item in response.json()["data"]]

    records = [
        {
            "id": i,
            "heading": chunk["heading"],
            "text": chunk["text"],
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(records)} chunk embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
