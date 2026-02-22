# debug_retrieval.py — safe version (no Qdrant lock conflict)
import httpx, json

ENDPOINT = "http://127.0.0.1:8080/diagnose"

test_cases = [
    {"symptoms": "сердце замирает, брадикардия, обморок, пульс 45", "expected_protocol": "АВ-блокада"},
    {"symptoms": "кровотечение из матки, болезненные менструации, боль в животе", "expected_protocol": "Эндометриоз/гинекология"},
    {"symptoms": "перелом позвоночника, боль в пояснице, травма при падении", "expected_protocol": "Травма позвоночника"},
    {"symptoms": "высокая температура, кашель, одышка, слабость", "expected_protocol": "Пневмония"},
]

for tc in test_cases:
    r = httpx.post(ENDPOINT, json={"symptoms": tc["symptoms"]}, timeout=30)
    data = r.json()
    print(f"\nQUERY: {tc['symptoms'][:60]}")
    print(f"EXPECTED: {tc['expected_protocol']}")
    for d in data["diagnoses"]:
        print(f"  rank={d['rank']} | {d['icd10_code']} | {d['diagnosis'][:50]}")
