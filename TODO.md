# TODO

## Done
- [x] First working Claude API call (`test_api.py`)
- [x] FAQ embeddings + retrieval (`embed_faq.py`, `ask_faq.py`)
- [x] Chatbot with history, streaming, error handling, out-of-scope check (`chatbot.py`)
- [x] Untrack venv, add `requirements.txt`
- [x] README

## Next
- [ ] **Security testing (prompt injection)**: try to make the bot ignore the FAQ
      or its instructions (e.g. "ignore previous instructions...", asking for the
      system prompt, off-topic requests), note what works, then add defenses
- [ ] **Regression test script**: a few in-scope and out-of-scope questions with
      expected behavior, run after every change
- [ ] **Browser UI**: simple web chat page instead of the terminal
      (e.g. Streamlit or Gradio for quickest start, or Flask/FastAPI + HTML)
