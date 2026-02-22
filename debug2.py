import json, os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path='.env')

client = OpenAI(
    base_url=os.getenv("LLM_HUB_URL"),
    api_key=os.getenv("LLM_API_KEY")
)

resp = client.chat.completions.create(
    model="oss-120b",
    messages=[
        {"role": "system", "content": 'Return only valid JSON: {"diagnoses": [{"rank": 1, "diagnosis": "test", "icd10_code": "A00", "explanation": "test"}]}'},
        {"role": "user", "content": "Кашель, температура 38"}
    ],
    max_tokens=2000,
    temperature=0.1
)

msg = resp.choices[0].message
raw = msg.reasoning_content or msg.content or ""

print(f"Raw length: {len(raw)}")
print(f"Last 50 chars repr: {repr(raw[-50:])}")

end = raw.rfind("}")
print(f"Last }} at: {end}")

depth = 0
start = -1
for i in range(end, -1, -1):
    if raw[i] == "}": depth += 1
    elif raw[i] == "{": depth -= 1
    if depth == 0:
        start = i
        break

candidate = raw[start:end+1]
print(f"Candidate length: {len(candidate)}")
print(f"First 50 repr: {repr(candidate[:50])}")
print(f"Last 50 repr:  {repr(candidate[-50:])}")

try:
    data = json.loads(candidate)
    print("✅ Parsed OK:", list(data.keys()))
except json.JSONDecodeError as e:
    print(f"❌ JSONDecodeError: {e}")
    print(f"   At pos {e.pos}, context: {repr(candidate[max(0,e.pos-20):e.pos+20])}")
    print(f"   Bytes: {candidate[max(0,e.pos-5):e.pos+5].encode('utf-8')}")

    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', candidate)
    try:
        data = json.loads(cleaned)
        print("✅ Works after stripping control chars!")
    except:
        print("❌ Still fails after cleaning")
        print("Full candidate:")
        print(candidate)
