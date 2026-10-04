# ai-security

A small RAG (retrieval-augmented generation) FAQ chatbot built on Claude.
It answers questions about a responsible-gaming FAQ (`data/faq.md`, in
Lithuanian) by finding the most relevant FAQ sections with Voyage embeddings
and passing them to Claude as context.

## How it works

1. `scripts/embed_faq.py` splits `data/faq.md` into chunks on `##` headings,
   embeds each chunk with `voyage-4-large`, and saves them to
   `data/faq_embeddings.json`.
2. At question time, the question is embedded the same way and compared to
   the stored chunks (cosine similarity). The top 3 chunks are sent to Claude
   along with the question.
3. If even the best match is weak (below `MIN_SIMILARITY = 0.3`), the bot
   says the FAQ doesn't cover it instead of calling Claude.

## Setup

```
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored):

```
ANTHROPIC_API_KEY=...
VOYAGE_API_KEY=...
```

`VOYAGE_API_KEY` is a MongoDB Atlas model API key, so embeddings are requested
from `https://ai.mongodb.com/v1/embeddings` rather than api.voyageai.com.

## Usage

| Command | What it does |
| --- | --- |
| `python test_api.py` | Sanity check that the Anthropic API key works |
| `python scripts/embed_faq.py` | Re-build `faq_embeddings.json` (run after editing `faq.md`) |
| `python scripts/ask_faq.py` | Simple one-question-at-a-time Q&A |
| `python scripts/chatbot.py` | Full chatbot: conversation history, streaming replies, error handling |

Type `quit` or `exit` to stop either interactive script.

## Chatbot settings

Tunable constants at the top of `scripts/chatbot.py`:

- `CHAT_MODEL`: Claude model used for answers
- `TOP_K`: how many FAQ chunks are retrieved per question (3)
- `MIN_SIMILARITY`: below this score the question is treated as out of scope (0.3)
- `MAX_HISTORY_MESSAGES`: cap on conversation history sent to Claude (20)

## Project layout

```
data/
  faq.md                 source FAQ
  faq_embeddings.json    precomputed chunk embeddings
scripts/
  embed_faq.py           build embeddings
  ask_faq.py             basic Q&A
  chatbot.py             full chatbot
test_api.py              API key check
requirements.txt
```
