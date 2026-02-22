import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(dotenv_path=".env")
client = OpenAI(base_url=os.getenv("LLM_HUB_URL"), api_key=os.getenv("LLM_API_KEY"))

resp = client.chat.completions.create(
    model="oss-120b",
    messages=[
        {"role": "system", "content": "Верни JSON: {\"diagnoses\":[{\"rank\":1,\"diagnosis\":\"...\",\"icd10_code\":\"X00\",\"explanation\":\"...\"}]}"},
        {"role": "user", "content": "Боль в животе, температура 38"}
    ],
    max_tokens=500, temperature=0.1
)
msg = resp.choices[0].message
print("content is None:", msg.content is None)
print("content:", repr(msg.content))
print("reasoning is None:", msg.reasoning_content is None)
