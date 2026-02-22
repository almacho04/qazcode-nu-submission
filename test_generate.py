import sys, json
sys.path.insert(0, '.')
from src.generator import generate_diagnosis

fake_chunks = [
    {
        "source_file": "HELLP-СИНДРОМ.pdf",
        "icd_codes": ["O14.2"],
        "text": "HELLP-синдром — жизнеугрожающее осложнение беременности. Симптомы: боль в правом подреберье, тошнота, повышение АД, тромбоцитопения, гемолиз."
    },
    {
        "source_file": "Преэклампсия.pdf",
        "icd_codes": ["O14.1"],
        "text": "Тяжёлая преэклампсия: АД выше 160/110, протеинурия, головная боль, нарушение зрения, боль в эпигастрии."
    },
    {
        "source_file": "Острый жировой гепатоз.pdf",
        "icd_codes": ["O75.0"],
        "text": "Острый жировой гепатоз беременных: тошнота, рвота, боль в животе, желтуха, печёночная недостаточность."
    }
]

query = "Пациентка 30 лет, беременность 28 недель. Жалобы: боль в правом подреберье, тошнота, рвота. АД 160/100. Отёки ног."
result = generate_diagnosis(query, fake_chunks)
print(json.dumps(result, ensure_ascii=False, indent=2))
