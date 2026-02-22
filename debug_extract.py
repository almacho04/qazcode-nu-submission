import sys, json
sys.path.insert(0, '.')

# Paste the exact raw text the model returns
raw = """cy (O75.0) less likely due to earlier gestational age and lack of jaundice, but still possible.

Provide JSON with rank, diagnosis, icd10_code, explanation.

Make sure JSON format exactly as required.
{
  "diagnoses": [
    {
      "rank": 1,
      "diagnosis": "HELLP-СИНДРОМ",
      "icd10_code": "O14.2",
      "explanation": "HELLP-синдром — жизнеугрожающее осложнение беременности. Симптомы: боль в правом подреберье, тошнота, повышение АД, тромбоцитопения, гемолиз."
    },
    {
      "rank": 2,
      "diagnosis": "Преэклампсия",
      "icd10_code": "O14.1",
      "explanation": "Тяжёлая преэклампсия: АД выше 160/110, протеинурия, головная боль, нарушение зрения, боль в эпигастрии."
    },
    {
      "rank": 3,
      "diagnosis": "Острый жировой гепатоз",
      "icd10_code": "O75.0",
      "explanation": "Острый жировой гепатоз беременных: тошнота, рвота, боль в животе, желтуха, печёночная недостаточность."
    }
  ]
}"""

# Step 1: find last }
end = raw.rfind('}')
print(f"Last }} at position: {end}, char: '{raw[end]}'")

# Step 2: walk backward
depth = 0
start = -1
for i in range(end, -1, -1):
    if raw[i] == '}': depth += 1
    elif raw[i] == '{': depth -= 1
    if depth == 0:
        start = i
        break

print(f"Opening {{ at position: {start}, char: '{raw[start]}'")
candidate = raw[start:end+1]
print(f"\nCandidate (first 100 chars): {repr(candidate[:100])}")
print(f"Candidate (last 100 chars):  {repr(candidate[-100:])}")

# Step 3: try to parse
try:
    result = json.loads(candidate)
    print("\n✅ JSON parsed successfully!")
    print(json.dumps(result, ensure_ascii=False, indent=2))
except json.JSONDecodeError as e:
    print(f"\n❌ JSON error: {e}")
    # Show exact character at error position
    print(f"Error at pos {e.pos}, char: {repr(candidate[e.pos-5:e.pos+5])}")
