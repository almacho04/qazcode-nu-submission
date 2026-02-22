import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_HUB_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
resp = client.chat.completions.create(
    model="oss-120b",
    messages=[{"role": "user", "content": "Назови столицу Казахстана одним словом."}],
    max_tokens=50
)

# Print EVERYTHING to see the structure
print("Full response:", resp)
print("---")
print("Choices:", resp.choices)
print("---")
msg = resp.choices[0].message
print("Message:", msg)
print("Content:", msg.content)
print("Role:", msg.role)

# Sometimes content is in tool_calls or reasoning
if hasattr(msg, 'reasoning_content'):
    print("Reasoning:", msg.reasoning_content)
