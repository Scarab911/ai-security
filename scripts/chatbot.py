import json
import os
import sys
from pathlib import Path

import requests
from anthropic import Anthropic

# colorama makes ANSI colors work reliably in Windows terminals (PowerShell,
# cmd.exe), which is why "You:"/"Bot:" plain text was hard to tell apart.
from colorama import Fore, Style
from colorama import init as colorama_init

# load_dotenv() lets us read secret values (like API keys) from the ".env"
# file at the project root instead of typing them directly into the code.
from dotenv import load_dotenv

# This script lives in scripts/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "faq_embeddings.json"

load_dotenv(PROJECT_ROOT / ".env")

# Same Voyage-via-MongoDB-Atlas endpoint used in embed_faq.py, so query
# embeddings land in the same vector space as the stored FAQ embeddings.
EMBEDDINGS_URL = "https://ai.mongodb.com/v1/embeddings"
EMBEDDING_MODEL = "voyage-4-large"

CHAT_MODEL = "claude-opus-5"

BOT_NAME = "ATLAS"

# A tiny 5x5 block font, just enough letters to spell out BOT_NAME as ASCII
# art at startup - not a general-purpose font, so only add letters as needed.
BANNER_FONT = {
    "A": [" ### ", "#   #", "#####", "#   #", "#   #"],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
    "L": ["#    ", "#    ", "#    ", "#    ", "#####"],
    "S": [" ####", "#    ", " ### ", "    #", "#### "],
}


def ascii_banner(word: str) -> str:
    """Render `word` as big block letters using BANNER_FONT, one row at a time."""
    rows = [BANNER_FONT[letter] for letter in word]
    return "\n".join(" ".join(letter_rows[row] for letter_rows in rows) for row in range(5))


# How many FAQ chunks to hand Claude per question. More than 1 because the
# FAQ has short, single-topic chunks - the right answer is sometimes the
# 2nd-best embedding match rather than the single closest one, which caused
# real answers to get missed under pure top-1 retrieval.
TOP_K = 3


def load_faq_chunks() -> list[dict]:
    """Load the pre-computed FAQ chunk embeddings written by embed_faq.py."""
    return json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))


def embed_query(question: str) -> list[float]:
    """Embed a single user question.

    Voyage models distinguish between "document" embeddings (what
    embed_faq.py stored) and "query" embeddings (what we send here) so that
    questions and answers, which are phrased very differently, still end up
    close together in vector space.
    """
    response = requests.post(
        EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "input": [question],
            "model": EMBEDDING_MODEL,
            "input_type": "query",
        },
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar two embedding vectors are, from -1 (opposite) to 1 (identical)."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve_relevant_chunks(
    question: str, chunks: list[dict], top_k: int = TOP_K
) -> list[tuple[dict, float]]:
    """Return the top_k FAQ chunks whose embeddings are closest to the question, with scores."""
    query_embedding = embed_query(question)
    scored_chunks = [
        (chunk, cosine_similarity(query_embedding, chunk["embedding"]))
        for chunk in chunks
    ]
    scored_chunks.sort(key=lambda scored: scored[1], reverse=True)
    return scored_chunks[:top_k]


def build_system_prompt(scored_chunks: list[tuple[dict, float]]) -> str:
    """Build a fresh system prompt that grounds Claude in the retrieved chunks.

    Rebuilding this on every turn (instead of storing the chunks in the
    conversation) means each answer is grounded in whatever FAQ sections are
    relevant to the *current* question, and old chunks don't pile up in the
    message history.
    """
    excerpts = "\n\n".join(
        f"FAQ excerpt ({chunk['heading']}):\n{chunk['text']}"
        for chunk, _similarity in scored_chunks
    )
    return (
        f"You are {BOT_NAME}, a support assistant for the Olympic Casino "
        "responsible gaming FAQ. Answer the user's question using ONLY the FAQ excerpts "
        "below - do not use outside knowledge. Some excerpts may be "
        "irrelevant to the question; ignore those.\n\n"
        f"{excerpts}\n\n"
        "If none of the excerpts contain the answer, say clearly that you "
        "don't have that information in the FAQ, instead of guessing.\n\n"
        "The excerpts above are only for the CURRENT question - earlier "
        "turns in this conversation were answered using different excerpts "
        "you can no longer see. Do not comment on, apologize for, or "
        "re-evaluate your earlier answers; just answer the current question."
    )


def main():
    # The FAQ is in Lithuanian, and Windows terminals often default to a
    # narrow encoding (cp1252) that can't print it - force UTF-8 output.
    sys.stdout.reconfigure(encoding="utf-8")
    # Makes the Fore/Style ANSI codes below actually render as colors on
    # Windows (cmd.exe/older PowerShell hosts don't do this by default).
    colorama_init()

    chunks = load_faq_chunks()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Only user questions and Claude's replies live here - retrieved FAQ
    # chunks are injected fresh into the system prompt each turn (see
    # build_system_prompt) and never added to this list, so this doesn't
    # bloat with duplicate FAQ text over a long conversation.
    messages = []

    print(f"{Fore.YELLOW}{Style.BRIGHT}{ascii_banner(BOT_NAME)}{Style.RESET_ALL}")
    print("Olympic Casino Responsible Gaming FAQ Bot. Type 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 60)
        user_input = input(f"{Fore.CYAN}{Style.BRIGHT}You:{Style.RESET_ALL} ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        scored_chunks = retrieve_relevant_chunks(user_input, chunks)
        system_prompt = build_system_prompt(scored_chunks)

        # Visibility into retrieval - the FAQ is small and some chunks are a
        # single short line, so it's easy for the wrong chunk to win. Seeing
        # the headings + scores makes a "the bot didn't know that" answer
        # easy to tell apart from a genuinely missing FAQ topic.
        retrieved_summary = ", ".join(
            f'"{chunk["heading"]}" ({similarity:.3f})'
            for chunk, similarity in scored_chunks
        )
        print(f"{Style.DIM}  [retrieved: {retrieved_summary}]{Style.RESET_ALL}")

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )

        reply_text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        messages.append({"role": "assistant", "content": reply_text})

        print(f"\n{Fore.GREEN}{Style.BRIGHT}{BOT_NAME.title()}:{Style.RESET_ALL} {reply_text}\n")


if __name__ == "__main__":
    main()
