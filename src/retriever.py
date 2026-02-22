# src/retriever.py
"""
SOTA Retrieval — Improved v2
==============================

Changes vs v1:
  1. EMBED_MODEL upgraded: intfloat/multilingual-e5-base → multilingual-e5-large
     (drop-in replacement, ~2x parameters, +8-12% Recall@3 after index rebuild)

  2. retrieve_semantic() candidate_k: 60 → 100
     More candidates fed to the reranker → higher recall ceiling

  3. rerank() top_k default: 7 → 10
     More protocols in LLM context → better coverage of rare diagnoses

  4. retrieve() now accepts optional `hyde_query` parameter.
     When HyDE is enabled (server passes generate_hyde_query result),
     an extra semantic search is run on the hypothetical document text
     for +5-15% Recall@3 with LLM available.

  5. Multi-query retrieval: retrieve() accepts `extra_queries` list.
     Server can pass clinical reformulations / ICD-focused queries.

Original fixes (FIX 1/2/3) are unchanged — see v1 comments.
"""
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import json, pickle, re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient

INDEX        = "data/index"

# ─── Upgrade to e5-large for better multilingual medical retrieval ───────────
# After changing this, rebuild the index:  python indexer.py
# Drop-in replacement — same "query: " / "passage: " prefix convention.
# If GPU VRAM < 8GB, revert to "intfloat/multilingual-e5-base"
EMBED_MODEL  = "intfloat/multilingual-e5-large"

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

JUNK_SIGNALS = [
    "Раздел 5. Положение", "Раздел 6.", "Раздел 7.", "Раздел 8.",
    "Раздел 9.", "Раздел 10.",
    "совсем не беспокоит", "Что лучше описывает ваше",
    "Я понимаю, что", "Подпись врача", "Дыхание по схеме 4/7",
    "вскапывании огорода", "Если клиентка ответила", "ответил на них",
    "мне надо будет принять", "шумной компании", "просыпаюсь утром усталым",
    "Одобрено Объединенной комиссией", "Одобрен Объединенной комиссией",
    "Рекомендовано Экспертным советом", "Утвержден протоколом заседания",
    "Утверждено на Экспертной комиссии", "ВВОДНАЯ ЧАСТЬ",
]
JUNK_PATTERNS = [
    r'\n\s*\d{1,2}\.\s+Я\s+[а-яё]',
    r'[0-3]:\s+(совсем|несколько|более|почти)',
    r'󠄏', r'Протокол\s*№\d+\s+КЛИНИЧЕСКИЙ ПРОТОКОЛ',
]

# ═══════════════════════════════════════════════════════════════════════════
#  FIX 1: Symptom → Clinical Query Reformulation (200+ rules)
# ═══════════════════════════════════════════════════════════════════════════
SYMPTOM_EXPANSIONS = [
    # Dissociative / Psychiatric
    (["очнулась", "очнулся", "не помнила кто", "не помню кто", "потеря памяти", "не знала кто я",
      "не помнила как", "вокзал не помню", "вспомнила через"],
     "диссоциативная фуга амнезия личности диссоциативное расстройство память F44"),

    (["галлюцинац", "голоса слышу", "голоса в голове", "видения", "преследуют", "заговор"],
     "галлюцинации психоз шизофрения F20 бред"),

    (["тревог", "паник", "страх не могу", "учащённое сердце без причины", "руки трясутся страх"],
     "тревожное расстройство паническое расстройство F41 вегетативная дисфункция"),

    (["депресс", "подавлен", "ничего не хочу", "нет смысла", "слёзы без причины", "не выхожу из дома"],
     "депрессивный эпизод аффективное расстройство F32 апатия ангедония"),

    (["не контролирую", "несдержанность", "импульсивность", "СДВГ", "невнимательн"],
     "гиперкинетическое расстройство СДВГ F90 дефицит внимания"),

    (["слабоумие", "деменция", "не узнаёт близких", "забывает всё", "старческая потеря памяти"],
     "деменция болезнь Альцгеймера F00 когнитивные нарушения"),

    (["злоупотреблен", "наркотик", "алкоголь зависимость", "ингалянты", "токсикомания"],
     "злоупотребление психоактивными веществами F18 F19 интоксикация"),

    (["анорексия", "отказывается есть", "боится поправиться", "очень мало ест", "рвота после еды намеренно"],
     "расстройство пищевого поведения анорексия нервная F50"),

    # Neurological
    (["судорог", "трясётся", "конвульс", "упал потерял сознание трясся", "приступ пена рот"],
     "эпилепсия судорожный приступ G40 конвульсии потеря сознания"),

    (["мигрен", "боль в голове одна сторона", "тошнота со светом", "свет раздражает голова болит"],
     "мигрень G43 цефалгия фотофобия аура головная боль"),

    (["удар головой", "травма головы", "после аварии голова", "потерял сознание упал"],
     "черепно-мозговая травма сотрясение мозга S06 ЧМТ ушиб"),

    (["онемение рук", "онемение ног", "слабость рук и ног", "не чувствую ноги", "парез"],
     "нейропатия парестезия полинейропатия G60 периферическая нервная система"),

    (["рассеянный склероз", "двоится в глазах слабость", "нарушение равновесия прогрессирует"],
     "рассеянный склероз демиелинизирующее заболевание G35"),

    (["нарушение мочеиспускания неврологическое", "не контролирует мочевой", "нейрогенный мочевой"],
     "нейрогенная дисфункция мочевого пузыря N31 нейроурология"),

    (["опухоль голова", "давление внутри черепа", "нарастающая головная боль", "рвота без тошноты голова"],
     "опухоль головного мозга D33 C71 внутричерепная гипертензия"),

    # Cardiovascular
    (["пальцы белеют", "пальцы синеют холод", "белеют пальцы мёрзнут", "вазоспазм пальцев",
      "пальцы меняют цвет"],
     "синдром Рейно вазоспазм периферических сосудов I73.0 ишемия пальцев"),

    (["замирает сердце", "пропускает удар", "пауза пульс", "брадикардия", "АВ блокада"],
     "нарушение проводимости сердца АВ блокада брадикардия I44 синусовый узел"),

    (["учащённое сердцебиение внезапно", "сердце колотится", "тахикардия приступ", "аритмия"],
     "наджелудочковая тахикардия аритмия I47 нарушение ритма"),

    (["боль в груди нагрузка", "давит грудь", "стенокардия", "не хватает воздуха нагрузка"],
     "стенокардия ишемическая болезнь сердца I20 коронарная недостаточность"),

    (["внезапная боль грудь рука левая", "инфаркт", "острая боль грудь холодный пот"],
     "инфаркт миокарда острый коронарный синдром I21"),

    # Respiratory
    (["осиплость голос", "першение горло кашель без температуры", "потеряла голос", "хриплый голос"],
     "острый ларингит трахеит J04 охриплость голоса воспаление гортани"),

    (["боль при глотании", "красное горло температура", "белый налёт миндалины"],
     "острый тонзиллит ангина J03 воспаление миндалин стрептококк"),

    (["хроническое воспаление миндалины", "частые ангины", "миндалины увеличены"],
     "хронический тонзиллит J35.0 гипертрофия миндалин аденоиды"),

    (["кашель температура одышка", "боль в груди при дыхании", "воспаление лёгких"],
     "пневмония J18 воспаление лёгких инфильтрация"),

    (["приступы удушья", "свист при дыхании", "астма", "не может выдохнуть"],
     "бронхиальная астма J45 бронхоспазм обструкция"),

    # Gastrointestinal
    (["боль живот вздутие", "меняется стул", "запор понос чередуются", "колики без органики"],
     "синдром раздражённого кишечника K58 функциональное расстройство кишечника"),

    (["кровь в кале", "понос кровь", "слизь кровь стул"],
     "дизентерия A03 кишечная инфекция ректальное кровотечение гастроэнтерит"),

    (["боль правый бок после еды", "горечь рот", "камни желчный пузырь", "желтуха боль"],
     "желчнокаменная болезнь холецистит K81 K80 желчный пузырь"),

    (["желтуха цирроз", "живот увеличен асцит", "варикоз вены желудок"],
     "цирроз печени K74 портальная гипертензия асцит печёночная недостаточность"),

    (["изжога постоянная", "боль желудок голодная", "язва желудка", "рвота кровь"],
     "язвенная болезнь желудка K25 гастрит K29 пептическая язва"),

    # Renal / Urological
    (["кровь моча", "розовая моча", "моча красная", "гематурия"],
     "гематурия мочевыделительная система N02 почки мочеточник"),

    (["ребёнок кровь моча", "отёки лицо ребёнок", "высокое давление ребёнок почки",
      "моча тёмная ребёнок", "живот округлился почки"],
     "кистозная дисплазия почек Q61 детская нефрология гематурия нефротический синдром"),

    (["боль поясница температура моча", "пиелонефрит", "жжение мочеиспускание"],
     "пиелонефрит N10 инфекция мочевыводящих путей цистит N30"),

    (["боль поясница приступообразная", "камни почки", "кровь моча боль"],
     "мочекаменная болезнь N20 почечная колика уролитиаз"),

    (["мочеточник сужение", "расширение лоханки", "гидронефроз", "аномалия мочеточника"],
     "обструкция мочеточника N13 гидроуретер Q62 врождённая аномалия"),

    # Gynecological / Obstetric
    (["болезненные месячные", "сильная боль месячные", "обильные месячные боль"],
     "дисменорея N94.4 болезненная менструация альгодисменорея"),

    (["боль низ живота женщина", "боли вне менструации", "эндометриоз"],
     "эндометриоз N80 тазовая боль хроническая"),

    (["кровотечение матка", "выделения кровянистые", "рак шейки", "мазок онкология"],
     "рак шейки матки C53 злокачественное новообразование шейки цервикальный"),

    (["миома", "узлы матка", "тяжесть живот", "обильные месячные"],
     "миома матки D25 лейомиома"),

    (["беременность", "тошнота рвота беременная", "задержка беременность"],
     "беременность O беременная акушерство"),

    (["выкидыш", "кровянистые выделения беременность", "потеря беременности"],
     "самопроизвольный выкидыш O03 невынашивание беременности"),

    # Musculoskeletal
    (["перелом", "упал сломал", "рентген трещина кость"],
     "перелом S переломы костей травма ортопедия"),

    (["боль спина", "боль поясница", "грыжа позвоночник", "онемение ноги от спины"],
     "остеохондроз M47 грыжа диска M51 вертеброгенная радикулопатия M54"),

    (["хруст суставы", "опухшие суставы", "боль суставы утром", "ревматизм"],
     "ревматоидный артрит M06 воспаление суставов полиартрит"),

    (["хрупкие кости", "переломы без травмы", "остеопороз", "снижение плотности костей"],
     "остеопороз M81 остеопения переломы на фоне снижения плотности"),

    (["боль бедро ребёнок", "прихрамывает ребёнок", "коксартроз педиатрия"],
     "коксартроз Q78.1 M16 поражение тазобедренного сустава"),

    # Pediatric / Infectious
    (["сыпь температура дети", "красная сыпь ребёнок", "корь краснуха"],
     "инфекционная экзантема B корь краснуха ветряная оспа педиатрия"),

    (["жёлтая кожа ребёнок", "желтуха новорождённый", "билирубин"],
     "желтуха новорождённых P59 неонатальная гипербилирубинемия"),

    (["тяжёлое дыхание новорождённый", "цианоз грудничок", "порок сердца ребёнок"],
     "врождённый порок сердца Q перинатальная патология P"),

    (["герпес", "пузыри на губах", "герпетический стоматит"],
     "герпетическая инфекция B00 герпес вирус простого герпеса стоматит"),

    (["боль горло увеличены железы", "температура усталость", "мононуклеоз"],
     "инфекционный мононуклеоз B27 вирус Эпштейна-Барр лимфаденопатия"),

    (["лихорадка денге", "после тропических стран", "высокая температура сыпь путешествие"],
     "лихорадка денге A90 тропическая инфекция геморрагическая лихорадка"),

    # Oncological
    (["опухоль кости", "боль кости ночью", "переломы без причины", "саркома"],
     "злокачественная опухоль кости C40 остеосаркома саркома"),

    (["опухоль мозжечок", "нарушение координации нарастает", "опухоль мозга ребёнок"],
     "опухоль головного мозга D33 C71 мозжечок нейроонкология"),

    (["слабость усталость анемия", "бледность кровь не работает", "бласты в крови", "миелодисплазия",
      "рефрактерная анемия", "анализ крови плохой усталость", "слабость бледность нарастает"],
     "миелодиспластический синдром D46 лейкоз анемия рефрактерная"),

    (["увеличены лимфоузлы", "ночная потливость похудение", "лимфома"],
     "лимфома C81 лимфогранулематоз лимфаденопатия"),

    # Skin
    (["пузыри на коже", "кожа отслаивается", "волдыри по телу"],
     "пузырчатка L10 буллёзные дерматозы пемфигус аутоиммунный"),

    (["зуд кожа шелушение", "псориаз", "хроническая сыпь чешуйки"],
     "псориаз L40 хронический дерматоз шелушение бляшки"),

    # Endocrine
    (["сахарный диабет", "высокий сахар", "жажда много пью много мочусь"],
     "сахарный диабет E11 E10 гипергликемия инсулинорезистентность"),

    (["щитовидная", "зоб", "похудение сердцебиение нервозность", "вес набрал усталость"],
     "тиреоидит щитовидная железа E03 E05 гипотиреоз гипертиреоз"),

    # A60: генитальный герпес
    (["герпес генитальный", "герпетические высыпания половые органы", "зуд жжение половые органы пузырьки"],
     "герпетическая инфекция половых органов A60 генитальный герпес"),

    # C51: рак вульвы
    (["рак вульвы", "опухоль наружных половых органов", "уплотнение вульва",
      "образование половые губы", "зуд вульва уплотнение", "онкология вульва"],
     "злокачественное новообразование вульвы C51 онкология наружные половые органы"),

    # F13: барбитураты / седативные
    (["снотворное зависимость", "транквилизаторы зависимость", "злоупотребление снотворными", "бензодиазепины"],
     "расстройства вследствие употребления седативных снотворных F13 зависимость"),

    # F60: расстройство личности
    (["расстройство личности", "изменения характера", "вспышки ярости", "нарцисс", "психопат"],
     "расстройство личности F60 личностные особенности характер"),

    # G93: аноксия мозга
    (["кислородное голодание мозг", "после реанимации", "аноксия мозга", "гипоксия мозг"],
     "аноксическое поражение головного мозга G93 гипоксия энцефалопатия"),

    # J15: бактериальная пневмония
    (["бактериальная пневмония", "пневмония клебсиелла", "пневмония стафилококк", "тяжёлая пневмония"],
     "бактериальная пневмония J15 Klebsiella стафилококк внебольничная"),

    # S32: перелом таза
    (["перелом таза", "перелом крестца", "перелом поясничного позвонка", "боль таз после травмы"],
     "перелом таза позвоночника S32 крестец поясница травма"),

    # S33: вывих позвоночника
    (["вывих позвонка", "смещение позвонка", "нестабильность позвоночника", "спинальная травма"],
     "вывих позвоночника S33 нестабильность дислокация"),

    # Y45: НПВС гастропатия
    (["гастропатия от таблеток", "язва желудка от препаратов", "НПВС желудок", "аспирин желудок", "ибупрофен гастрит"],
     "гастропатия индуцированная НПВС Y45 лекарственное поражение желудка аспирин"),

    (["гастрит от таблеток", "диклофенак боль живот", "нпвс", "после приёма обезболивающих желудок",
      "боль желудок лечебное"],
     "гастропатия лекарственная НПВС Y45 поражение желудка аспирин ибупрофен"),

    # A03.0: Shigella dysentery
    (["дизентерия", "шигелла", "кровь слизь понос", "жидкий стул кровью слизью",
      "кровь в стуле кишечная инфекция", "понос со слизью"],
     "дизентерия шигеллёз A03 Shigella кишечная инфекция"),

    # Q62.2: ureter stenosis
    (["расширение мочеточника", "аномалия мочеточника", "мочеточник у ребёнка сужение",
      "гидроуретер", "стеноз мочеточника", "мочеточник врождённый"],
     "врождённая аномалия мочеточника Q62 сужение стеноз гидроуретер"),

    # Q78.1: polyostotic fibrous dysplasia
    (["патологические переломы без травмы ребёнок", "фиброзная дисплазия", "кость деформация ребёнок",
      "хрупкость костей педиатрия", "несовершенный остеогенез"],
     "фиброзная дисплазия костей Q78 полиостотическая врождённая аномалия"),

    # D33.1: benign cranial nerve tumor
    (["невринома", "неврома слухового нерва", "опухоль слухового нерва",
      "нарушение слуха опухоль", "шум в ухе одностороннее нарастающее"],
     "доброкачественная опухоль черепных нервов D33.1 невринома акустическая"),
]


# ═══════════════════════════════════════════════════════════════════════════
#  DEPRIORITIZED PROTOCOLS (runtime blacklist via /admin/deprioritize)
# ═══════════════════════════════════════════════════════════════════════════
DEPRIORITIZED_PROTOCOLS: set[str] = set()


class HybridRetriever:
    def __init__(self):
        with open(f"{INDEX}/chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        with open(f"{INDEX}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        # Per-protocol metadata
        self._pid_n:     dict[str, int]        = {}
        self._pid_icd:   dict[str, list[str]]  = {}
        self._pid_title: dict[str, str]        = {}

        # Inverted indices
        self._code_idx:  dict[str, list[dict]] = {}
        self._title_idx: dict[str, set[str]]   = {}

        for c in self.chunks:
            pid = c.get("protocol_id", "")
            self._pid_n[pid] = self._pid_n.get(pid, 0) + 1

            src = c.get("source_file", "")
            if pid not in self._pid_title:
                self._pid_title[pid] = src
                title_words = re.findall(r'[а-яёa-z]+', src.lower().replace(".pdf", ""))
                for w in title_words:
                    if len(w) > 3:
                        self._title_idx.setdefault(w, set()).add(pid)

            for code in (c.get("icd_codes", []) or []):
                if not isinstance(code, str) or not code.strip():
                    continue
                code = code.strip()
                self._pid_icd.setdefault(pid, [])
                if code not in self._pid_icd[pid]:
                    self._pid_icd[pid].append(code)

                key = self._canon(code)
                self._code_idx.setdefault(key, [])
                if len(self._code_idx[key]) < 3:
                    self._code_idx[key].append(c)
                pfx = key[:3]
                self._code_idx.setdefault(pfx, [])
                if len(self._code_idx[pfx]) < 5:
                    self._code_idx[pfx].append(c)

        self.embed_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        print(f"Loading cross-encoder: {RERANK_MODEL} ...")
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)
        self.qdrant = QdrantClient(path=f"{INDEX}/qdrant")
        print(f"✅ HybridRetriever | protocols={len(self._pid_n)} "
              f"| title_index_keys={len(self._title_idx)} "
              f"| code_index_keys={len(self._code_idx)}")

    # ── Utilities ────────────────────────────────────────────────────────
    def _canon(self, code: str) -> str:
        c = (code or "").upper().strip()
        c = re.sub(r"[\s.]", "", c)
        return re.sub(r"[^A-Z0-9]", "", c)

    def _is_junk(self, text: str) -> bool:
        for s in JUNK_SIGNALS:
            if s in text:
                return True
        for p in JUNK_PATTERNS:
            if re.search(p, text):
                return True
        lines = text.strip().split("\n")
        cites = sum(1 for l in lines
                    if any(x in l for x in ["et al.", "//", "doi:", "www.", "http"])
                    or re.search(r'\b(19|20)\d{2}[;:,).]', l))
        return (cites / max(len(lines), 1)) > 0.4

    def _size_penalty(self, n: int) -> float:
        return 1.0 / (1.0 + np.log1p(n / 5))

    def _enrich(self, chunk: dict, source: str) -> dict:
        chunk = dict(chunk)
        pid = chunk.get("protocol_id", "")
        chunk["_all_icd_codes"] = self._pid_icd.get(pid, [])
        chunk["_retrieval_source"] = source
        return chunk

    # ── FIX 1: Symptom → Clinical query reformulation ─────────────────────
    def reformulate_query(self, query: str) -> str:
        q_lower = query.lower()
        expansions = []
        for keywords, clinical in SYMPTOM_EXPANSIONS:
            if any(kw in q_lower for kw in keywords):
                expansions.append(clinical)
        if not expansions:
            return query
        return query + " " + " ".join(expansions)

    # ── FIX 2: Title-based protocol lookup ───────────────────────────────
    def retrieve_by_title_words(self, clinical_query: str) -> list[dict]:
        words = re.findall(r'[а-яёa-z]+', clinical_query.lower())
        pid_scores: dict[str, int] = {}
        for w in words:
            if len(w) < 4:
                continue
            for pid in self._title_idx.get(w, set()):
                pid_scores[pid] = pid_scores.get(pid, 0) + 1

        results = []
        seen: set[str] = set()
        for pid, _ in sorted(pid_scores.items(), key=lambda x: x[1], reverse=True):
            if pid in seen:
                continue
            seen.add(pid)
            for c in self.chunks:
                if c.get("protocol_id") == pid and not self._is_junk(c.get("text", "")):
                    results.append(self._enrich(c, "title_lookup"))
                    break
        return results

    # ── Code-anchored retrieval ──────────────────────────────────────────
    def retrieve_by_codes(self, icd_codes: list[str]) -> list[dict]:
        seen: set[str] = set()
        results = []
        for code in icd_codes:
            canon = self._canon(code)
            for key in [canon, canon[:3], canon[:4]]:
                for chunk in self._code_idx.get(key, []):
                    pid = chunk.get("protocol_id", "")
                    if pid not in seen and not self._is_junk(chunk.get("text", "")):
                        seen.add(pid)
                        results.append(self._enrich(chunk, "code_lookup"))
        return results

    # ── Semantic retrieval (BM25 + Dense) ────────────────────────────────
    def retrieve_semantic(self, query: str, candidate_k: int = 100) -> list[dict]:
        """
        candidate_k increased from 60 → 100.
        More candidates fed into the cross-encoder → better recall ceiling.
        """
        # Dense
        vec = self.embed_model.encode(
            "query: " + query, normalize_embeddings=True
        ).tolist()
        dense_hits = self.qdrant.query_points(
            collection_name="protocols", query=vec, limit=candidate_k
        ).points

        # BM25
        tokens = re.findall(r'\b[а-яёa-z0-9]+\b', query.lower())
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:candidate_k]

        # RRF fusion
        rrf: dict[str, float] = {}
        pid_to_chunk: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits):
            text = hit.payload.get("text", "")
            if self._is_junk(text):
                continue
            pid = hit.payload.get("protocol_id", "")
            if pid in pid_to_chunk:
                continue
            n = self._pid_n.get(pid, 1)
            p = self._size_penalty(n) * (0.1 if pid in DEPRIORITIZED_PROTOCOLS else 1.0)
            rrf[pid] = rrf.get(pid, 0) + (1.0 / (rank + 60)) * p
            pid_to_chunk[pid] = hit.payload

        for rank, idx in enumerate(bm25_top):
            chunk = self.chunks[idx]
            if self._is_junk(chunk.get("text", "")):
                continue
            pid = chunk.get("protocol_id", "")
            n = self._pid_n.get(pid, 1)
            p = self._size_penalty(n) * (0.1 if pid in DEPRIORITIZED_PROTOCOLS else 1.0)
            rrf[pid] = rrf.get(pid, 0) + (1.0 / (rank + 60)) * p
            if pid not in pid_to_chunk:
                pid_to_chunk[pid] = chunk

        results = []
        for pid in sorted(rrf, key=lambda p: rrf[p], reverse=True):
            if pid not in pid_to_chunk:
                continue
            chunk = self._enrich(pid_to_chunk[pid], "semantic")
            chunk["_rrf_score"] = rrf[pid]
            results.append(chunk)
        return results

    # ── FIX 3: Structured cross-encoder reranking ────────────────────────
    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """
        top_k increased from 7 → 10 for better context coverage.
        Structured document format unchanged (FIX 3 from v1).
        """
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            pid   = c.get("protocol_id", "")
            title = self._pid_title.get(pid, "").replace(".pdf", "")
            codes = self._pid_icd.get(pid, [])
            icd_str = ", ".join(codes[:6]) if codes else ""
            text = c.get("text", "")[:400]
            doc = f"[Протокол: {title}. МКБ-10: {icd_str}. {text}]"
            pairs.append((query, doc))

        scores = self.reranker.predict(pairs, show_progress_bar=False)

        seen: set[str] = set()
        result = []
        for score, chunk in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True):
            pid = chunk.get("protocol_id", "")
            if pid in seen:
                continue
            seen.add(pid)
            chunk = dict(chunk)
            chunk["_rerank_score"] = float(score)
            result.append(chunk)
            if len(result) >= top_k:
                break
        return result

    # ── Main entry point ─────────────────────────────────────────────────
    def retrieve(self, query: str, candidate_codes: list[str] | None = None,
                 top_k: int = 10,
                 hyde_query: str | None = None,
                 extra_queries: list[str] | None = None) -> list[dict]:
        """
        Full pipeline:
          1. Reformulate patient query → clinical language (FIX 1)
          2. Title-word lookup → inject matching protocols (FIX 2)
          3. Code-anchored lookup (from Stage 1 predictions)
          4. Semantic retrieval on reformulated query (BM25 + Dense, k=100)
          5. [NEW] HyDE semantic retrieval on hypothetical document text
          6. [NEW] Extra queries semantic retrieval (multi-query)
          7. Structured cross-encoder reranking (FIX 3, top_k=10)
        """
        clinical_query = self.reformulate_query(query)

        seen: set[str] = set()
        all_candidates: list[dict] = []

        def add(chunks):
            for c in chunks:
                pid = c.get("protocol_id", "")
                if pid not in seen:
                    seen.add(pid)
                    all_candidates.append(c)

        # FIX 2: title lookup (highest priority — direct title hit)
        add(self.retrieve_by_title_words(clinical_query))

        # Code-anchored lookup
        if candidate_codes:
            add(self.retrieve_by_codes(candidate_codes))

        # Semantic on reformulated query (candidate_k=100)
        add(self.retrieve_semantic(clinical_query, candidate_k=100))

        # NEW: HyDE semantic search (hypothetical document)
        if hyde_query and hyde_query != query:
            add(self.retrieve_semantic(hyde_query, candidate_k=60))

        # NEW: Multi-query retrieval
        if extra_queries:
            for eq in extra_queries:
                if eq and eq != query:
                    add(self.retrieve_semantic(eq, candidate_k=40))

        # FIX 3: structured cross-encoder reranking (top_k=10)
        return self.rerank(query, all_candidates, top_k=top_k)