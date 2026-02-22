# src/generator.py
"""
Generation Pipeline — Improved v2
===================================
Changes vs v1:
  1. FIX — Smart fallback code sorting: uses EXACT Stage 1 ICD match first,
     then ICD description token-overlap scoring to pick best sub-code
     (was: alphabetical sort → always picked I44.0 instead of I44.1)
  2. FIX — Multi-model LLM fallback chain: tries cheaper models before
     giving up so 402/credit errors don't silently kill both stages
  3. NEW — HyDE query: generate_hyde_query() produces a hypothetical
     protocol excerpt from symptoms; retriever.retrieve() accepts it as
     an optional extra query for much better dense recall
  4. NEW — ICD_DESCRIPTIONS dict used in fallback code ranking (no API)
  5. IMPROVED — local_predict_codes_full() returns both deduplicated list
     and the full matched code set for exact ranking
"""

import os, json, re, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=".env")

client = OpenAI(
    base_url=os.getenv("LLM_HUB_URL", "https://hub.qazcode.ai/v1"),
    api_key=os.getenv("LLM_API_KEY"),
)

# ─── Model chain: primary → cheapest fallbacks ───────────────────────────────
# The API key only allows access to 'oss-120b' (no "openai/" prefix).
# FALLBACK_MODELS is intentionally empty — only add models your key can access.
PRIMARY_MODEL   = os.getenv("LLM_MODEL", "oss-120b")
# PRIMARY_MODEL   = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

FALLBACK_MODELS: list[str] = []


# ════════════════════════════════════════════════════════════════════════════
#  ICD sub-code → Russian description
#  Used by the fallback code sorter to pick the best sub-code WITHOUT LLM.
#  Extend freely — the more entries, the better the no-LLM accuracy.
# ════════════════════════════════════════════════════════════════════════════
ICD_DESCRIPTIONS: dict[str, str] = {
    # AV conduction block
    "I44.0": "предсердно-желудочковая блокада первой степени AV блокада 1 степени удлинение PQ интервала",
    "I44.1": "предсердно-желудочковая блокада второй степени AV блокада 2 степени Мобитц выпадение QRS",
    "I44.2": "предсердно-желудочковая блокада полная третья степень полная поперечная блокада",
    "I44.3": "другая неуточнённая предсердно-желудочковая блокада AV",
    "I45.2": "двухпучковая блокада бифасцикулярная блокада ножек",
    "I49.5": "синдром слабости синусового узла СССУ брадикардия паузы",
    # Tachyarrhythmia
    "I47.1": "наджелудочковая тахикардия пароксизм",
    "I47.2": "желудочковая тахикардия пароксизм",
    "I48.0": "фибрилляция предсердий пароксизмальная мерцательная аритмия",
    "I49.0": "фибрилляция трепетание желудочков",
    # IBS
    "K58.0": "синдром раздражённого кишечника с диареей понос",
    "K58.9": "синдром раздражённого кишечника без диареи запор",
    "K58":   "синдром раздражённого кишечника СРК функциональное расстройство",
    # Cholecystitis
    "K81.0": "острый холецистит желчный пузырь воспаление острое",
    "K81.1": "хронический холецистит желчный пузырь воспаление хроническое",
    "K80.0": "камни желчного пузыря с острым холециститом ЖКБ",
    "K80.1": "камни желчного пузыря с другим холециститом",
    "K80.2": "камни желчного пузыря без холецистита желчнокаменная болезнь",
    # Dysentery
    "A03.0": "шигеллёз вызванный Shigella dysenteriae дизентерия кровь слизь",
    "A03.9": "шигеллёз неуточнённый дизентерия",
    "A02.0": "сальмонеллёзный энтерит гастроэнтерит сальмонелла",
    "A02.1": "сальмонеллёзная септицемия сепсис",
    "A02.2": "локализованная сальмонеллёзная инфекция",
    # Dysmenorrhea / PMS
    "N94.3": "предменструальный синдром ПМС боли перед менструацией",
    "N94.4": "первичная дисменорея болезненные менструации альгодисменорея",
    "N94.5": "вторичная дисменорея эндометриоз",
    "N94.6": "дисменорея неуточнённая",
    # Laryngitis
    "J04.0": "острый ларингит воспаление гортани осиплость охриплость",
    "J04.1": "острый трахеит воспаление трахеи",
    "J04.2": "острый ларинготрахеит",
    "J02.0": "стрептококковый фарингит",
    "J02.9": "острый фарингит неуточнённый боль горло",
    # Cervical cancer
    "C53.0": "злокачественное новообразование эндоцервикса шейка матки",
    "C53.1": "злокачественное новообразование экзоцервикса шейка матки",
    "C53.8": "злокачественное новообразование шейки матки перекрёстное поражение",
    "C53.9": "злокачественное новообразование шейки матки неуточнённое рак",
    # Pemphigus / bullous
    "L10.0": "пузырчатка обыкновенная аутоиммунный буллёзный дерматоз",
    "L10.1": "пузырчатка вегетирующая",
    "L12.0": "буллёзный пемфигоид волдыри кожа",
    # Personality disorders
    "F60.0": "параноидное расстройство личности подозрительность",
    "F60.1": "шизоидное расстройство личности замкнутость",
    "F60.2": "диссоциальное расстройство личности антисоциальное",
    "F60.3": "эмоционально неустойчивое расстройство личности импульсивность",
    # Dissociative
    "F44.0": "диссоциативная амнезия потеря памяти психогенная",
    "F44.1": "диссоциативная фуга бродяжничество потеря личности вокзал",
    "F44.8": "другие диссоциативные расстройства конверсионные",
    # Depression
    "F32.0": "лёгкий депрессивный эпизод подавленность",
    "F32.1": "умеренный депрессивный эпизод депрессия средняя",
    "F32.2": "тяжёлый депрессивный эпизод суицидальные мысли",
    # Epilepsy
    "G40.0": "локализованная идиопатическая эпилепсия фокальные судороги",
    "G40.3": "генерализованная идиопатическая эпилепсия тонико-клонические",
    "G40.9": "эпилепсия неуточнённая судороги приступ",
    # NSAID gastropathy
    "Y45.1": "НПВС нестероидные противовоспалительные ацетилсалициловая аспирин гастропатия",
    "Y45.2": "салицилаты лекарственная язва желудка",
    "Y45.3": "другие НПВС нестероидные поражение желудка ибупрофен диклофенак",
    # Renal cysts
    "Q61.0": "одиночная киста почки врождённая",
    "Q61.3": "поликистоз почек аутосомно-рецессивный дети",
    "Q61.4": "кистозная дисплазия почки",
    # Ureter
    "Q62.0": "врождённый гидронефроз расширение лоханки",
    "Q62.2": "врождённое сужение мочеточника стеноз",
    "N13.5": "перегиб стриктура мочеточника без гидронефроза обструкция",
    # Inhalants
    "F18.0": "острая интоксикация ингалянтами клей растворитель нюхает",
    "F18.1": "употребление ингалянтов с вредными последствиями токсикомания",
    "F18.2": "синдром зависимости от ингалянтов хроническая токсикомания",
    "F18.8": "другие психические расстройства ингалянты",
    # Hallucinogens
    "F16.0": "острая интоксикация галлюциногенами ЛСД психоделики",
    "F16.1": "употребление галлюциногенов с вредными последствиями",
    "F16.2": "синдром зависимости от галлюциногенов",
    # Osteoporosis
    "M81.0": "постменопаузальный остеопороз переломы женщины климакс",
    "M81.1": "остеопороз после удаления яичников",
    # Raynaud
    "I73.0": "синдром Рейно вазоспазм пальцы белеют синеют",
    # Post-COVID
    "F06.6": "органическое эмоционально лабильное расстройство постковид хроническая усталость",
    "F41.2": "смешанное тревожное и депрессивное расстройство",
    # Bone dysplasia
    "Q78.1": "полиостотическая фиброзная дисплазия костей врождённая патологические переломы",
}


def score_code_vs_query(code: str, query: str) -> float:
    """
    Token-overlap score between an ICD code's clinical description
    and the patient query. Fast, no model or API needed.
    """
    desc = ICD_DESCRIPTIONS.get(code, "")
    if not desc:
        return 0.0
    q_tokens = set(re.findall(r'[а-яёa-z0-9]+', query.lower()))
    d_tokens = set(re.findall(r'[а-яёa-z0-9]+', desc.lower()))
    if not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / (len(d_tokens) + 1)


# ════════════════════════════════════════════════════════════════════════════
#  LOCAL STAGE 1 — keyword → ICD code mapping (zero API calls)
# ════════════════════════════════════════════════════════════════════════════
KEYWORD_ICD_MAP: list[tuple[list[str], list[str]]] = [
    # Neurological / Psychiatric
    (["очнулась", "не помнила кто", "не помню кто", "потеря памяти", "вокзал не помню",
      "вспомнила через", "не знала кто я", "не помнила как"],
     ["F44.1", "F44.0", "F44.8"]),
    (["галлюцинац", "голоса в голове", "голоса слышу", "видения", "преследуют", "заговор"],
     ["F20.0", "F20.9", "F23.0"]),
    (["тревог", "паник", "страх навязчив", "руки трясутся страх"],
     ["F41.0", "F41.1", "F40.0"]),
    (["депресс", "подавлен", "ничего не хочу", "нет смысла", "слёзы без причины"],
     ["F32.0", "F32.1", "F32.2"]),
    (["судорог", "трясётся", "конвульс", "упал потерял сознание трясся", "приступ пена рот"],
     ["G40.0", "G40.3", "G40.9"]),
    (["мигрен", "боль голова одна сторона", "тошнота свет", "фотофобия голова"],
     ["G43.0", "G43.1"]),
    (["удар головой", "травма головы", "после аварии голова", "потерял сознание упал"],
     ["S06.0", "S06.3", "S09.9"]),
    (["слабоумие", "деменция", "не узнаёт близких", "забывает всё"],
     ["F00.0", "F00.9"]),
    (["рассеянный склероз", "двоится в глазах слабость", "нарушение равновесия прогрессирует"],
     ["G35"]),
    (["инсульт", "афазия", "паралич лица", "онемение половина тела"],
     ["I63.9", "I64"]),
    # Cardiovascular
    (["замирает сердце", "пропускает удар", "пауза пульс", "брадикард", "редкий пульс",
      "остановка сердца ощущение", "ав блокад", "av блокада", "блокада сердца",
      "нарушение проводимости сердца"],
     ["I44.1", "I44.2", "I44.0", "I49.5"]),
    (["учащённое сердцебиение", "сердце колотится", "тахикард", "аритм"],
     ["I47.1", "I48.0", "I49.0"]),
    (["боль грудь нагрузка", "давит грудь", "стенокардия"],
     ["I20.0", "I20.9", "I25.1"]),
    (["инфаркт", "острая боль грудь холодный пот", "боль грудь рука левая"],
     ["I21.0", "I21.9"]),
    (["гипертон", "давление высокое", "гипертензия"],
     ["I10", "I11.0"]),
    (["пальцы белеют", "пальцы синеют холод", "белеют пальцы мёрзнут", "рейно", "вазоспазм пальцев"],
     ["I73.0"]),
    # Respiratory
    (["осиплость", "потеряла голос", "хрипота", "першение горло", "ларингит"],
     ["J04.0", "J04.1", "J04.2"]),
    (["боль горло", "красное горло температура", "белый налёт миндалины", "ангин"],
     ["J03.9", "J35.0"]),
    (["кашель температура одышка", "боль грудь дыхание", "воспаление лёгких", "пневмони"],
     ["J18.1", "J15.0", "J18.9"]),
    (["астм", "свист дыхание", "удушье приступ", "бронхоспазм"],
     ["J45.0", "J45.9"]),
    # GI
    (["боль живот вздутие", "меняется стул", "запор понос чередуются", "срк"],
     ["K58.0", "K58.9"]),
    (["кровь кал", "понос кровь", "слизь кровь стул", "дизентери", "шигелл"],
     ["A03.0", "A03.9"]),
    (["боль правый бок еда", "горечь рот", "камни желчный", "желтуха боль бок"],
     ["K81.0", "K81.1", "K80.2"]),
    (["цирроз", "асцит", "живот увеличен"],
     ["K74.5", "K74.6"]),
    (["изжога", "боль желудок голодная", "язва желудок"],
     ["K25.0", "K29.7", "K21.0"]),
    (["нпвс", "аспирин желудок", "ибупрофен желудок", "диклофенак желудок", "от таблеток желудок"],
     ["Y45.1", "Y45.2", "Y45.3"]),
    # Renal
    (["кровь моча", "гематурия", "розовая моча", "моча красная"],
     ["N02.9", "Q61.4", "N20.0"]),
    (["пиелонефрит", "жжение мочеиспускание", "боль поясница температура моча"],
     ["N10", "N11.0"]),
    (["мочекаменн", "камни почки", "почечная колика"],
     ["N20.0", "N20.1"]),
    (["кисты почки", "поликистоз", "кистозная почка"],
     ["Q61.3", "Q61.4", "Q61.0"]),
    (["мочеточник сужение", "гидронефроз", "расширение лоханки", "аномалия мочеточника"],
     ["Q62.2", "Q62.0", "N13.5"]),
    # Gynecological
    (["болезненные месячные", "боль месячные", "дисменорея", "альгодисменорея"],
     ["N94.4", "N94.5", "N94.6"]),
    (["предменструальный", "пмс", "перед месячными"],
     ["N94.3"]),
    (["эндометриоз", "боль низ живота женщина вне месячных"],
     ["N80.0", "N80.9"]),
    (["рак шейки матки", "онкология шейки", "мазок онкология", "кровотечение шейка"],
     ["C53.0", "C53.8", "C53.9"]),
    (["миома", "узлы матка", "обильные месячные", "лейомиома"],
     ["D25.0", "D25.1", "D25.9"]),
    (["беременность", "тошнота рвота беременная", "задержка"],
     ["O21.0", "O26.6"]),
    (["выкидыш", "самопроизвольный аборт"],
     ["O03.0", "O03.9"]),
    # Musculoskeletal
    (["перелом", "рентген трещина кость"],
     ["S32.2", "S32.0", "S42.0"]),
    (["боль спина", "поясничная боль", "остеохондроз"],
     ["M54.5", "M51.1"]),
    (["артрит", "воспаление суставов", "ревматоидный"],
     ["M06.0", "M05.0"]),
    (["остеопороз", "хрупкие кости переломы без травмы"],
     ["M81.0", "M81.1"]),
    # Pediatric
    (["ребёнок кровь моча", "отёки лицо ребёнок", "моча тёмная ребёнок"],
     ["Q61.4", "N04.9", "N00.9"]),
    (["гиперактивность", "сдвг", "невнимательность ребёнка"],
     ["F90.0", "F90.9"]),
    (["патологические переломы ребёнок", "фиброзная дисплазия"],
     ["Q78.1"]),
    # Infectious
    (["туберкулёз", "кашель кровь длительный", "ночная потливость похудение кашель"],
     ["A15.0", "A15.3"]),
    (["герпес", "пузыри губах", "герпетический стоматит"],
     ["B00.2", "A60.0"]),
    (["лихорадка денге", "тропическая лихорадка", "сыпь после тропических стран"],
     ["A90", "A91"]),
    # Oncological
    (["опухоль головного мозга", "глиом", "нарастающая головная боль рвота"],
     ["C71.0", "C71.9", "D33.0", "D33.1"]),
    (["лимфом", "увеличены лимфоузлы ночная потливость"],
     ["C81.0", "C82.0"]),
    (["миелодиспласт", "рефрактерная анемия", "бласты крови"],
     ["D46.2", "D46.4"]),
    # Endocrine
    (["диабет", "высокий сахар", "жажда много пью много мочусь"],
     ["E11.9", "E10.9"]),
    (["щитовидная", "гипотиреоз", "гипертиреоз", "зоб"],
     ["E03.9", "E05.0"]),
    # Skin
    (["псориаз", "хроническая сыпь чешуйки бляшки"],
     ["L40.0", "L40.9"]),
    (["пузыри на коже", "пемфигус", "буллёзн", "пузырчатка", "волдыри тело"],
     ["L10.0", "L10.1", "L12.0"]),
    # Personality
    (["расстройство личности", "параноидн", "подозрительн характер", "вспышки ярости", "психопат"],
     ["F60.0", "F60.1", "F60.2", "F60.3"]),
    # Inhalants
    (["ингалянт", "клей нюхает", "растворитель нюхает", "ацетон нюхает", "токсикомания ингалянт"],
     ["F18.0", "F18.1", "F18.2", "F18.8"]),
    # Hallucinogens
    (["галлюциноген", "психоделик", "лсд", "грибы наркотик"],
     ["F16.0", "F16.1", "F16.2"]),
    # Post-COVID
    (["постковид", "после ковида", "долгий ковид", "хроническая усталость после covid"],
     ["F06.6", "F41.2", "F32.0"]),
    # AV block synonyms
    (["блокада ножки пучка", "неполная блокада", "бифасцикулярная"],
     ["I45.2", "I44.1", "I44.0"]),
    (["слабость синусового узла", "сссу", "синусовая брадикардия выраженная"],
     ["I49.5", "I44.1"]),
]


def local_predict_codes_full(query: str) -> tuple[list[str], set[str]]:
    """
    Keyword-based ICD prediction.
    Returns (deduped_codes, full_code_set).
    full_code_set contains ALL matched codes for exact-match ranking in fallback.
    """
    q = query.lower()
    scores: dict[str, int] = {}
    full_code_set: set[str] = set()

    for keywords, codes in KEYWORD_ICD_MAP:
        for kw in keywords:
            if kw.lower() in q:
                for code in codes:
                    full_code_set.add(code)
                    scores[code] = scores.get(code, 0) + 1

    if not scores:
        return [], set()

    sorted_codes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for code, _ in sorted_codes:
        prefix = code[:3]
        if prefix not in seen:
            seen.add(prefix)
            result.append(code)
        if len(result) >= 5:
            break
    return result, full_code_set


def local_predict_codes(query: str) -> list[str]:
    codes, _ = local_predict_codes_full(query)
    return codes


# ════════════════════════════════════════════════════════════════════════════
#  LLM HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _try_llm_call(model: str, messages: list, max_tokens: int = 500) -> str | None:
    """Single LLM call. Returns content string or None on any failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        err = str(e)
        if "402" in err or "credit" in err.lower() or "balance" in err.lower():
            print(f"[INFO] {model} unavailable (402/credit)")
        elif "404" in err or "not found" in err.lower():
            print(f"[INFO] {model} not found")
        else:
            print(f"[WARN] {model}: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
#  HYDE — Hypothetical Document Embedding
#  Generates a synthetic protocol excerpt to improve dense retrieval recall.
#  Pass the result as `hyde_query` to retriever.retrieve().
# ════════════════════════════════════════════════════════════════════════════
HYDE_SYSTEM = (
    "Ты — медицинский эксперт. По симптомам пациента напиши 2-3 предложения, "
    "как звучал бы соответствующий раздел казахстанского клинического протокола. "
    "Включи: вероятный код МКБ-10, основные диагностические критерии, ключевые термины. "
    "Отвечай только на русском языке, кратко и клинически точно."
)


def generate_hyde_query(query: str) -> str:
    """
    HyDE: returns a hypothetical protocol excerpt for the symptoms.
    Falls back to original query if all LLMs unavailable.
    """
    messages = [
        {"role": "system", "content": HYDE_SYSTEM},
        {"role": "user", "content": f"Симптомы пациента: {query}"},
    ]
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        result = _try_llm_call(model, messages, max_tokens=250)
        if result:
            print(f"[INFO] HyDE generated via {model}")
            return result
    print("[INFO] HyDE fallback → original query")
    return query


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Predict candidate ICD codes
# ════════════════════════════════════════════════════════════════════════════
CODE_PREDICTION_SYSTEM = """Ты — клинический кодировщик МКБ-10.
По описанию пациента определи 5 наиболее вероятных диагнозов с кодами МКБ-10.
Верни ТОЛЬКО JSON (без markdown):
{
  "key_symptoms": ["симптом 1", "симптом 2"],
  "candidates": [
    {"rank": 1, "diagnosis": "Название", "icd10": "X00.0", "rationale": "1 предложение"},
    {"rank": 2, "diagnosis": "Название", "icd10": "X00.1", "rationale": "1 предложение"},
    {"rank": 3, "diagnosis": "Название", "icd10": "X00.2", "rationale": "1 предложение"},
    {"rank": 4, "diagnosis": "Название", "icd10": "X00.3", "rationale": "1 предложение"},
    {"rank": 5, "diagnosis": "Название", "icd10": "X00.4", "rationale": "1 предложение"}
  ]
}"""


def predict_candidate_codes(query: str) -> tuple[list[str], dict]:
    """
    LLM Stage 1 with multi-model fallback chain, then local keyword fallback.
    Returns (codes_list, prediction_dict).
    prediction_dict["full_codes"] is a set of ALL predicted codes (for exact matching).
    """
    messages = [
        {"role": "system", "content": CODE_PREDICTION_SYSTEM},
        {"role": "user", "content": f"Пациент: {query}"},
    ]

    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        raw = _try_llm_call(model, messages, max_tokens=500)
        if raw is None:
            continue
        prediction = _extract_json(raw)
        if prediction and "candidates" in prediction:
            codes = [
                c.get("icd10", "").strip()
                for c in prediction["candidates"]
                if c.get("icd10", "").strip()
            ]
            if codes:
                prediction["full_codes"] = set(codes)
                print(f"[INFO] Stage 1 codes via {model}: {codes[:3]}")
                return codes, prediction

    # All LLMs failed → local fallback
    local_codes, full_code_set = local_predict_codes_full(query)
    if local_codes:
        print(f"[INFO] Stage 1 local codes: {local_codes}")
    return local_codes, {"full_codes": full_code_set}


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Generate final diagnosis
# ════════════════════════════════════════════════════════════════════════════
DIAGNOSIS_SYSTEM = """Ты — клинический ассистент казахстанской системы здравоохранения.

По жалобам пациента и клиническим протоколам РК определи ТОП-3 дифференциальных диагноза.

АЛГОРИТМ:
1. Сопоставь симптомы с диагностическими критериями каждого протокола
2. Выбери наиболее подходящий протокол
3. Из его «Кодов МКБ-10» выбери НАИБОЛЕЕ СПЕЦИФИЧНЫЙ код, соответствующий описанным симптомам

ПРАВИЛА:
- icd10_code ДОЛЖЕН быть из раздела «Коды МКБ-10» указанного протокола
- Выбирай КОНКРЕТНЫЙ подтип (например I44.1 а не I44, K58.0 а не K58)
- explanation: 2-3 предложения своими словами (не цитируй протокол)
- Ровно 3 диагноза, только JSON

{
  "diagnoses": [
    {"rank": 1, "diagnosis": "Название", "icd10_code": "X00.0", "explanation": "Обоснование"},
    {"rank": 2, "diagnosis": "Название", "icd10_code": "X00.1", "explanation": "Обоснование"},
    {"rank": 3, "diagnosis": "Название", "icd10_code": "X00.2", "explanation": "Обоснование"}
  ]
}"""


def build_diagnosis_prompt(query: str, context_chunks: list[dict],
                           stage1_prediction: dict | None = None) -> str:
    prelim = ""
    if stage1_prediction and stage1_prediction.get("candidates"):
        lines = [
            f"  - {c.get('diagnosis','')} [{c.get('icd10','')}]: {c.get('rationale','')}"
            for c in stage1_prediction["candidates"][:3]
        ]
        prelim = "ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ:\n" + "\n".join(lines) + "\n\n"

    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source_file", "").replace(".pdf", "")
        text = (chunk.get("text", "") or "")[:3500]
        all_codes = chunk.get("_all_icd_codes") or chunk.get("icd_codes", []) or []
        icd_str = ", ".join(str(c) for c in all_codes if c) or "не указаны"
        score = chunk.get("_rerank_score")
        hint = f" [score={score:.3f}]" if score else ""
        context += (
            f"══ Протокол {i}: {source}{hint} ══\n"
            f"Коды МКБ-10: {icd_str}\n"
            f"{text}\n\n"
        )

    return (
        f"ЖАЛОБЫ ПАЦИЕНТА:\n{query}\n\n"
        f"{prelim}"
        f"КЛИНИЧЕСКИЕ ПРОТОКОЛЫ РК:\n{context}"
        "Определи ТОП-3 дифференциальных диагноза. "
        "Используй КОНКРЕТНЫЙ код МКБ-10 из раздела «Коды МКБ-10» наиболее подходящего протокола. "
        "Выбирай наиболее специфичный подтип кода. "
        "Объяснение — своими словами. Верни только JSON."
    )


def generate_diagnosis(query: str, context_chunks: list[dict],
                       stage1_prediction: dict | None = None,
                       _retries: int = 2) -> dict:
    prompt = build_diagnosis_prompt(query, context_chunks, stage1_prediction)
    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        for attempt in range(_retries):
            raw = _try_llm_call(model, messages, max_tokens=1200)
            if raw is None:
                break  # try next model
            result = _extract_json(raw)
            if result and "diagnoses" in result:
                print(f"[INFO] Stage 2 diagnosis via {model}")
                return _ensure_three(result)
            if "429" in str(raw):
                wait = 15 * (attempt + 1)
                print(f"[RATE LIMIT] {model} — waiting {wait}s...")
                time.sleep(wait)

    return _smart_fallback(context_chunks, stage1_prediction, query)


# ════════════════════════════════════════════════════════════════════════════
#  SMART FALLBACK — no LLM, improved code ranking
# ════════════════════════════════════════════════════════════════════════════
def _smart_fallback(chunks: list[dict], stage1: dict | None, query: str = "") -> dict:
    """
    Extract diagnoses from reranked chunks without LLM.

    Code ranking priority (for each chunk):
      1. Exact Stage 1 match  (e.g. predicted "I44.1" and chunk has "I44.1")
      2. Prefix Stage 1 match + ICD description token-overlap score
      3. Has a dot (specific sub-code) vs generic code
      4. Alphabetical (deterministic tiebreak)

    This fixes the original alphabetical sort bug where I44.0 was always
    picked over I44.1 even when Stage 1 clearly predicted I44.1.
    """
    diagnoses: list[dict] = []
    rank = 1
    seen_codes: set[str] = set()

    # Collect full codes from Stage 1
    stage1_full: set[str] = set()
    stage1_prefix: set[str] = set()
    if stage1:
        for code in (stage1.get("full_codes") or set()):
            stage1_full.add(code)
            stage1_prefix.add(code[:3])
        for c in (stage1.get("candidates") or []):
            code = c.get("icd10", "").strip()
            if code:
                stage1_full.add(code)
                stage1_prefix.add(code[:3])

    for chunk_idx, chunk in enumerate(chunks):
        if rank > 3:
            break

        source = chunk.get("source_file", "").replace(".pdf", "")
        all_codes = chunk.get("_all_icd_codes") or chunk.get("icd_codes", []) or []
        all_codes = [str(c).strip() for c in all_codes if str(c).strip()]

        def code_sort_key(c: str) -> tuple:
            exact     = c in stage1_full           # highest priority
            prefix    = c[:3] in stage1_prefix
            desc_sc   = score_code_vs_query(c, query) if query else 0.0
            has_dot   = "." in c
            return (not exact, not prefix, -desc_sc, not has_dot, c)

        ordered = sorted(all_codes, key=code_sort_key)

        for code in ordered:
            if rank > 3:
                break
            if code in seen_codes:
                continue
            seen_codes.add(code)
            diagnoses.append({
                "rank": rank,
                "diagnosis": source,
                "icd10_code": code,
                "explanation": (
                    f"Протокол «{source}» является наиболее релевантным по данным симптомам "
                    f"(reranker position {chunk_idx + 1})."
                ),
            })
            rank += 1

    return _ensure_three({"diagnoses": diagnoses})


# ════════════════════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════════════════════
def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if "```" in text:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            text = m.group(1).strip()
    if not text.startswith("{"):
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            text = m.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return {"diagnoses": [
                {**d, "rank": d.get("rank", i + 1)}
                for i, d in enumerate(data[:3])
            ]}
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _ensure_three(result: dict) -> dict:
    diags = (result or {}).get("diagnoses", []) or []
    diags = diags[:3]
    while len(diags) < 3:
        diags.append({
            "rank": len(diags) + 1,
            "diagnosis": "—",
            "icd10_code": "",
            "explanation": "",
        })
    for i, d in enumerate(diags, 1):
        d["rank"] = i
        d.setdefault("diagnosis", "—")
        d.setdefault("icd10_code", "")
        d.setdefault("explanation", "")
    result["diagnoses"] = diags
    return result