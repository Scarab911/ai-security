import os

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
