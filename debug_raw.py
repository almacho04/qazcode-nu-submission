import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")
client = OpenAI(
    base_url=os.getenv("LLM_HUB_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

resp = client.chat.completions.create(
    model="oss-120b",
    messages=[
        {"role": "system", "content": "Ты — медицинский ассистент. Верни ТОП-3 диагноза как JSON."},
        {"role": "user", "content": "Кашель, температура 38, боль в горле"}
    ],
    max_tokens=1000,
    temperature=0.1
)

msg = resp.choices[0].message

print("=== CONTENT ===")
print(repr(msg.content))

print("\n=== REASONING_CONTENT (full) ===")
print(repr(msg.reasoning_content))

print("\n=== REASONING PRINTED RAW ===")
print(msg.reasoning_content)
