# debug_corpus_codes.py
import sys, json
sys.path.insert(0, "src")

with open("data/index/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Show icd_codes for the protocols that keep winning
keywords = [
    "паллиативн", "нижней части спины", "постковидный",
    "нарушение сна", "азартным"
]
seen = set()
for c in chunks:
    sf = c.get("source_file", "").lower()
    pid = c.get("protocol_id", "")
    codes = c.get("icd_codes", [])
    if pid in seen:
        continue
    for kw in keywords:
        if kw in sf:
            seen.add(pid)
            print(f"PROTOCOL: {c.get('source_file','')}")
            print(f"  icd_codes: {codes}")
            print(f"  pid: {pid}")
            break
