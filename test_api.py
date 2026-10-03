import os

# load_dotenv() lets us read secret values (like API keys) from a local ".env"
# file instead of typing them directly into the code. This keeps secrets out
# of source control.
from dotenv import load_dotenv

# Anthropic is the official Python client for talking to Claude models.
from anthropic import Anthropic

# Reads the ".env" file in this folder and loads its values as environment
# variables (e.g. ANTHROPIC_API_KEY=...) so os.environ can see them below.
load_dotenv()

# Create a client object that knows how to authenticate with the Anthropic
# API. os.environ["ANTHROPIC_API_KEY"] fetches the key we just loaded from
# the .env file.
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Send a request to Claude and wait for its reply.
# - model: which Claude model to use
# - max_tokens: the maximum length of the reply (in tokens, roughly word pieces)
# - messages: the conversation so far, as a list of {"role": ..., "content": ...}
#   dicts. Here we send a single user message asking Claude to say hello.
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

# Claude's reply comes back as a list of "content blocks" (response.content).
# A block can be different types (e.g. "text"), so we loop through them and
# print out only the text ones.
for block in response.content:
    if block.type == "text":
        print(block.text)
