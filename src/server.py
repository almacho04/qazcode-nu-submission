# server.py
"""
Server — Improved v2
=====================
(Backend logic unchanged — UI only enhanced)
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.retriever import HybridRetriever, DEPRIORITIZED_PROTOCOLS
from src.generator import (
    predict_candidate_codes,
    generate_diagnosis,
    generate_hyde_query,
    local_predict_codes,
)

retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = HybridRetriever()
    yield


app = FastAPI(lifespan=lifespan)


class Req(BaseModel):
    symptoms: str
    model_config = {"extra": "ignore"}


# ─── Main endpoint ────────────────────────────────────────────────────────────
@app.post("/diagnose")
async def diagnose(req: Req):
    loop = asyncio.get_event_loop()

    (candidate_codes, stage1_prediction), hyde_query = await asyncio.gather(
        loop.run_in_executor(None, predict_candidate_codes, req.symptoms),
        loop.run_in_executor(None, generate_hyde_query, req.symptoms),
    )

    local_codes = local_predict_codes(req.symptoms)
    extra_queries = []
    if local_codes:
        extra_queries.append(f"клинический протокол {' '.join(local_codes[:3])}")

    chunks = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(
            query=req.symptoms,
            candidate_codes=candidate_codes,
            top_k=10,
            hyde_query=hyde_query,
            extra_queries=extra_queries,
        ),
    )

    result = await loop.run_in_executor(
        None, lambda: generate_diagnosis(req.symptoms, chunks, stage1_prediction)
    )
    return JSONResponse(content=result)


# ─── Debug endpoint ───────────────────────────────────────────────────────────
@app.post("/diagnose_debug")
async def diagnose_debug(req: Req, top_k: int = Query(10, ge=1, le=30)):
    loop = asyncio.get_event_loop()

    (candidate_codes, stage1_prediction), hyde_query = await asyncio.gather(
        loop.run_in_executor(None, predict_candidate_codes, req.symptoms),
        loop.run_in_executor(None, generate_hyde_query, req.symptoms),
    )

    local_codes = local_predict_codes(req.symptoms)
    extra_queries = []
    if local_codes:
        extra_queries.append(f"клинический протокол {' '.join(local_codes[:3])}")

    chunks = await loop.run_in_executor(
        None,
        lambda: retriever.retrieve(
            query=req.symptoms,
            candidate_codes=candidate_codes,
            top_k=top_k,
            hyde_query=hyde_query,
            extra_queries=extra_queries,
        ),
    )

    result = await loop.run_in_executor(
        None, lambda: generate_diagnosis(req.symptoms, chunks, stage1_prediction)
    )

    return JSONResponse(
        content={
            "stage1_candidate_codes": candidate_codes,
            "stage1_prediction": {
                k: v
                for k, v in (stage1_prediction or {}).items()
                if k != "full_codes"
            },
            "hyde_query": hyde_query,
            "extra_queries": extra_queries,
            "retrieved": [
                {
                    "rank": i + 1,
                    "protocol_id": c.get("protocol_id"),
                    "source_file": c.get("source_file"),
                    "icd_codes_all": c.get("_all_icd_codes"),
                    "rerank_score": c.get("_rerank_score"),
                    "retrieval_source": c.get("_retrieval_source"),
                    "text_preview": (c.get("text") or "")[:200],
                }
                for i, c in enumerate(chunks)
            ],
            "result": result,
        }
    )


# ─── Retrieval-only ──────────────────────────────────────────────────────────
@app.post("/retrieve_only")
async def retrieve_only(
    req: Req,
    top_k: int = Query(10, ge=1, le=30),
    use_hyde: bool = Query(False),
):
    loop = asyncio.get_event_loop()
    candidate_codes, _ = await loop.run_in_executor(None, predict_candidate_codes, req.symptoms)

    hyde_query = None
    if use_hyde:
        hyde_query = await loop.run_in_executor(None, generate_hyde_query, req.symptoms)

    chunks = await loop.run_in_executor(
        None, lambda: retriever.retrieve(req.symptoms, candidate_codes, top_k, hyde_query=hyde_query)
    )

    return JSONResponse(
        content={
            "predicted_codes": candidate_codes,
            "hyde_query": hyde_query,
            "retrieved": [
                {
                    "rank": i + 1,
                    "protocol_id": c.get("protocol_id"),
                    "source_file": c.get("source_file"),
                    "icd_codes_all": c.get("_all_icd_codes"),
                    "rerank_score": round(c.get("_rerank_score", 0), 4),
                    "retrieval_source": c.get("_retrieval_source"),
                }
                for i, c in enumerate(chunks)
            ],
        }
    )


# ─── Admin: manage deprioritized protocols ────────────────────────────────────
@app.post("/admin/deprioritize/{protocol_id}")
async def deprioritize(protocol_id: str):
    DEPRIORITIZED_PROTOCOLS.add(protocol_id)
    return {"deprioritized": list(DEPRIORITIZED_PROTOCOLS)}


@app.delete("/admin/deprioritize/{protocol_id}")
async def undeprioritize(protocol_id: str):
    DEPRIORITIZED_PROTOCOLS.discard(protocol_id)
    return {"deprioritized": list(DEPRIORITIZED_PROTOCOLS)}


@app.get("/admin/deprioritized")
async def list_deprioritized():
    return {"deprioritized": list(DEPRIORITIZED_PROTOCOLS)}


# ─── UI ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(
        content=r"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AI Диагностика — Протоколы РК</title>
  <style>
    :root{
      --bg:#0b1220;
      --panel: rgba(255,255,255,.07);
      --panel2: rgba(255,255,255,.04);
      --text:#e5e7eb;
      --muted:#a8b0bf;
      --border:rgba(255,255,255,.12);
      --accent:#60a5fa;
      --good:#22c55e;
      --warn:#f59e0b;
      --vio:#a78bfa;
      --bad:#fb7185;
      --shadow: 0 14px 40px rgba(0,0,0,.35);
      --r:18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:var(--sans);
      color:var(--text);
      background:
        radial-gradient(900px 520px at 20% 10%, rgba(96,165,250,.26), transparent 62%),
        radial-gradient(900px 520px at 82% 18%, rgba(34,197,94,.14), transparent 62%),
        radial-gradient(1000px 700px at 50% 100%, rgba(167,139,250,.12), transparent 60%),
        var(--bg);
      min-height:100vh;
    }
    .wrap{max-width:1180px;margin:0 auto;padding:24px 16px 56px}
    .top{
      display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;
      margin-bottom:14px;
    }
    .brand{
      display:flex;gap:10px;align-items:center;
      padding:10px 12px;border-radius:999px;border:1px solid var(--border);
      background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.04));
      box-shadow: var(--shadow);
    }
    .logo{
      width:34px;height:34px;border-radius:12px;
      background: radial-gradient(circle at 30% 30%, #93c5fd, #2563eb);
      display:grid;place-items:center;font-weight:900;
      color:#fff;
    }
    .brand b{font-size:14px}
    .brand .sub{font-size:12px;color:var(--muted)}
    .chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
    .chip{
      font-size:12px;padding:7px 10px;border-radius:999px;border:1px solid var(--border);
      background: rgba(255,255,255,.05); color: var(--text);
    }
    .hero{
      margin:14px 0 16px;
      border:1px solid var(--border);
      border-radius:var(--r);
      background: linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
      box-shadow: var(--shadow);
      padding:16px 16px;
      display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-end;
    }
    .hero h1{margin:0;font-size:22px}
    .hero p{margin:8px 0 0;color:var(--muted);max-width:760px;font-size:13px;line-height:1.35}
    .grid{display:grid;grid-template-columns: 1fr 1fr; gap:16px; align-items:start;}
    @media(max-width:980px){.grid{grid-template-columns:1fr}}
    .panel{
      border:1px solid var(--border);
      border-radius:var(--r);
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .hd{
      padding:12px 14px;
      border-bottom:1px solid var(--border);
      display:flex;align-items:center;justify-content:space-between;gap:10px;
      background: rgba(255,255,255,.04);
      font-weight:900;
      font-size:13px;
    }
    .hd small{font-weight:700;color:var(--muted)}
    .bd{padding:14px}
    textarea{
      width:100%;
      min-height:190px;
      padding:12px 12px;
      border-radius:14px;
      border:1px solid var(--border);
      background: rgba(0,0,0,.24);
      color: var(--text);
      outline:none;
      resize:vertical;
      font-size:14px;
      line-height:1.4;
    }
    textarea:focus{
      border-color: rgba(96,165,250,.55);
      box-shadow: 0 0 0 4px rgba(96,165,250,.16);
    }
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:12px}
    .btn{
      appearance:none;border:none;
      padding:10px 14px;border-radius:12px;
      font-weight:900;font-size:14px;cursor:pointer;
      display:inline-flex;align-items:center;gap:8px;
    }
    .primary{color:#06101f;background: linear-gradient(180deg,#93c5fd,#60a5fa)}
    .ghost{color:var(--text);background:transparent;border:1px solid var(--border)}
    .btn:disabled{opacity:.55;cursor:not-allowed}
    .hint{
      margin-top:10px;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.03);
      color: var(--muted);
      font-size:12px;
      line-height:1.35;
    }
    #err{
      display:none;margin-top:10px;padding:10px 12px;border-radius:12px;
      border:1px solid rgba(251,113,133,.42);background:rgba(251,113,133,.10);
      color:#fecdd3;font-size:13px
    }
    #err.show{display:block}

    /* Results cards */
    .card{
      border:1px solid var(--border);
      border-radius:14px;
      padding:14px;
      background: rgba(255,255,255,.04);
      margin-bottom:10px;
      box-shadow: 0 10px 24px rgba(0,0,0,.18);
    }
    .card.r1{border-left:4px solid var(--good)}
    .card.r2{border-left:4px solid var(--warn)}
    .card.r3{border-left:4px solid var(--vio)}
    .tags{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}
    .tag{
      font-size:11px;padding:4px 9px;border-radius:999px;border:1px solid var(--border);
      background: rgba(255,255,255,.05);
    }
    .icd{font-family:var(--mono);color:#dbeafe;border-color:rgba(96,165,250,.35);background:rgba(96,165,250,.12)}
    .muted{color:var(--muted)}
    .empty{color:var(--muted);font-size:14px;padding:10px 0}
    .mono{font-family:var(--mono)}
    .lat{font-size:12px;color:var(--muted);margin-top:8px}

    /* Debug */
    details{
      margin-top:12px;
      border:1px solid rgba(255,255,255,.10);
      border-radius:14px;
      background: rgba(255,255,255,.03);
      padding:10px 12px;
    }
    summary{cursor:pointer;font-weight:900}
    .kv{display:grid;grid-template-columns:170px 1fr;gap:8px 12px;margin-top:10px;font-size:12px}
    @media(max-width:520px){.kv{grid-template-columns:1fr}}
    .kv div{color:var(--muted)}
    .kv b{color:var(--text)}
    table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
    th,td{border-bottom:1px solid rgba(255,255,255,.10);padding:8px 6px;text-align:left;vertical-align:top}
    th{color:var(--muted);font-weight:800}
    pre{
      white-space:pre-wrap;word-break:break-word;margin:10px 0 0;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(0,0,0,.22);
      padding:10px 12px;border-radius:14px;font-size:12px;
    }
    .minirow{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .mini{
      padding:7px 10px;border-radius:10px;font-size:12px;font-weight:900;
      border:1px solid var(--border);background:rgba(255,255,255,.05);color:var(--text);
      cursor:pointer
    }
    .mini:hover{border-color:rgba(96,165,250,.45)}

    /* Loading */
    .status{
      display:none;
      margin-top:10px;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.03);
      color: var(--muted);
      align-items:center;justify-content:space-between;gap:10px;
      font-size:12px;
    }
    .status.show{display:flex}
    .spin{
      width:16px;height:16px;border-radius:999px;
      border:2px solid rgba(255,255,255,.22);
      border-top-color: rgba(147,197,253,.95);
      animation: sp 0.85s linear infinite;
    }
    @keyframes sp{to{transform:rotate(360deg)}}

    /* History */
    .hist{display:none;margin-top:10px}
    .hchips{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
    .hchip{
      cursor:pointer;user-select:none;
      font-size:12px;padding:7px 10px;border-radius:999px;border:1px solid var(--border);
      background: rgba(255,255,255,.05);color: var(--text);
    }
    .toast{
      position:fixed;right:16px;bottom:16px;z-index:50;
      padding:10px 12px;border-radius:14px;border:1px solid var(--border);
      background:rgba(17,24,39,.88);color:var(--text);box-shadow:var(--shadow);
      display:none;max-width:340px;font-size:12px
    }
    .toast.show{display:block}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="brand">
      <div class="logo">Rx</div>
      <div>
        <b>AI Диагностика</b><div class="sub">Протоколы РК • ICD-10 • HyDE • Rerank</div>
      </div>
    </div>
    <div class="chips">
      <span class="chip">Endpoint: <span class="mono">/diagnose_debug</span></span>
      <span class="chip">Top-K: <span class="mono">10</span></span>
      <span class="chip">Ctrl+Enter</span>
    </div>
  </div>

  <div class="hero">
    <div>
      <h1>🏥 Клинический ассистент (demo)</h1>
      <p>
        Введите симптомы/анамнез. Интерфейс покажет: кандидаты МКБ-10 (Stage 1), HyDE-запрос,
        найденные протоколы (retrieval evidence) и финальные ТОП-3 диагнозов.
      </p>
    </div>
    <div class="chips">
      <span class="chip">⏱ тайминги</span>
      <span class="chip">📚 evidence</span>
      <span class="chip">🧾 raw JSON</span>
    </div>
  </div>

  <div class="grid">
    <!-- LEFT -->
    <div class="panel">
      <div class="hd">📋 Симптомы пациента <small id="pill">готово</small></div>
      <div class="bd">
        <textarea id="q" placeholder="Например: Мужчина 55 лет. Боль за грудиной 20 минут, одышка, холодный пот, иррадиация в левую руку..."></textarea>

        <div id="err"></div>

        <div class="status" id="status">
          <div style="display:flex;align-items:center;gap:10px">
            <div class="spin"></div>
            <div>
              <div style="font-weight:900;color:var(--text)">Анализирую…</div>
              <div class="muted">Stage1 → HyDE → Retrieval → Diagnosis</div>
            </div>
          </div>
          <div class="mono" id="timer">0.0s</div>
        </div>

        <div class="row">
          <button class="btn primary" id="runBtn" onclick="go()">🔍 Диагностировать</button>
          <button class="btn ghost" onclick="clearAll()">🧹 Очистить</button>
          <button class="btn ghost" onclick="toggleHistory()">🕘 История</button>
        </div>

        <div class="hist" id="hist">
          <div class="muted" style="font-size:12px;margin-top:10px">Последние запросы:</div>
          <div class="hchips" id="histList"></div>
          <div class="minirow">
            <button class="mini" onclick="clearHistory()">🗑 Очистить историю</button>
          </div>
        </div>

        <div class="hint">
          Подсказка: добавьте возраст/пол/длительность/иррадиацию/температуру/анамнез — качество retrieval обычно растёт.
        </div>
      </div>
    </div>

    <!-- RIGHT -->
    <div class="panel">
      <div class="hd">🎯 Результаты <small id="lat">—</small></div>
      <div class="bd">
        <div id="out"><div class="empty">Введите симптомы и нажмите «Диагностировать».</div></div>

        <details id="dbg" style="display:none">
          <summary>🧩 Debug: Stage 1 / HyDE / Evidence / Raw</summary>

          <div class="kv" id="kv"></div>

          <details open style="margin-top:12px">
            <summary>📚 Retrieval evidence</summary>
            <table>
              <thead>
                <tr>
                  <th>#</th><th>Protocol</th><th>Source</th><th>Score</th><th>Mode</th>
                </tr>
              </thead>
              <tbody id="tbl"></tbody>
            </table>
          </details>

          <div class="minirow">
            <button class="mini" onclick="copyJSON()">📋 Copy JSON</button>
            <button class="mini" onclick="copyTopICD()">📋 Copy ICD#1</button>
          </div>

          <pre id="raw"></pre>
        </details>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const qEl = document.getElementById('q');
const outEl = document.getElementById('out');
const errEl = document.getElementById('err');
const runBtn = document.getElementById('runBtn');
const latEl = document.getElementById('lat');
const pill = document.getElementById('pill');
const statusEl = document.getElementById('status');
const timerEl = document.getElementById('timer');
const dbg = document.getElementById('dbg');
const kv = document.getElementById('kv');
const tbl = document.getElementById('tbl');
const rawEl = document.getElementById('raw');
const toastEl = document.getElementById('toast');
const histBox = document.getElementById('hist');
const histList = document.getElementById('histList');

let LAST = null;
let tick = null;

function esc(s){
  return (s||'').toString()
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'","&#039;");
}
function toast(msg){
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(()=>toastEl.classList.remove('show'), 1400);
}
function setError(m){
  if(!m){ errEl.classList.remove('show'); errEl.textContent=''; return; }
  errEl.textContent = m;
  errEl.classList.add('show');
}
function setLoading(on){
  if(on){
    statusEl.classList.add('show');
    runBtn.disabled = true;
    pill.textContent = 'в работе…';
  }else{
    statusEl.classList.remove('show');
    runBtn.disabled = false;
    pill.textContent = 'готово';
  }
}

function renderResult(d){
  const res = d?.result || d;
  const ds = (res && res.diagnoses) ? res.diagnoses.slice().sort((a,b)=>(a.rank||0)-(b.rank||0)) : [];
  if(!ds.length){
    outEl.innerHTML = '<div class="empty">Нет результатов.</div>';
    return;
  }
  const labels = ['','✅ Высокая вероятность','⚠️ Средняя вероятность','🟣 Низкая вероятность'];
  outEl.innerHTML = ds.map((x,i)=>{
    const r = Number(x.rank || (i+1));
    const rc = (r===1?'r1':(r===2?'r2':'r3'));
    return `<div class="card ${rc}">
      <div style="font-weight:900">${'#'+r+' '} ${esc(x.diagnosis||'—')}</div>
      <div class="tags">
        <span class="tag">${labels[r]||''}</span>
        <span class="tag icd">МКБ-10: ${esc(x.icd10_code||'—')}</span>
      </div>
      <div class="muted" style="margin-top:10px;font-size:13px;line-height:1.35">${esc(x.explanation||'')}</div>
    </div>`;
  }).join('');
}

function renderDebug(d){
  dbg.style.display = 'block';
  kv.innerHTML = '';

  const codes = (d.stage1_candidate_codes || []).join(', ');
  const hyde = (d.hyde_query || '').toString();
  const extra = (d.extra_queries || []).join(' | ');

  const items = [
    ['Stage1 codes', codes || '—'],
    ['HyDE query', hyde ? (hyde.length > 260 ? hyde.slice(0,260)+'…' : hyde) : '—'],
    ['Extra queries', extra || '—'],
  ];

  for(const [k,v] of items){
    const a = document.createElement('div'); a.textContent = k;
    const b = document.createElement('b'); b.textContent = v;
    kv.appendChild(a); kv.appendChild(b);
  }

  tbl.innerHTML = '';
  const rows = d.retrieved || [];
  rows.forEach(r=>{
    const score = (r.rerank_score==null ? '—' : Number(r.rerank_score).toFixed(4));
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono">${esc(String(r.rank||''))}</td>
      <td class="mono">${esc(String(r.protocol_id||''))}</td>
      <td>${esc(String(r.source_file||''))}</td>
      <td class="mono">${esc(score)}</td>
      <td class="mono">${esc(String(r.retrieval_source||''))}</td>
    `;
    tbl.appendChild(tr);
  });

  rawEl.textContent = JSON.stringify(d, null, 2);
}

async function copyJSON(){
  if(!LAST){ toast('Нет данных'); return; }
  await navigator.clipboard.writeText(JSON.stringify(LAST, null, 2));
  toast('JSON скопирован');
}
async function copyTopICD(){
  const icd = LAST?.result?.diagnoses?.find(x=>x.rank===1)?.icd10_code
           || LAST?.result?.diagnoses?.[0]?.icd10_code;
  if(!icd){ toast('Нет ICD'); return; }
  await navigator.clipboard.writeText(icd);
  toast('ICD#1 скопирован');
}

/* History */
const HKEY = 'diag_history_v2';
function getHistory(){
  try{ return JSON.parse(localStorage.getItem(HKEY) || '[]'); }catch(e){ return []; }
}
function saveHistory(arr){ localStorage.setItem(HKEY, JSON.stringify(arr)); }
function pushHistory(text){
  const v = text.trim(); if(!v) return;
  let h = getHistory();
  h = h.filter(x => x !== v);
  h.unshift(v);
  h = h.slice(0, 8);
  saveHistory(h);
  renderHistory();
}
function renderHistory(){
  const h = getHistory();
  histList.innerHTML = '';
  if(!h.length){
    histList.innerHTML = '<span class="muted" style="font-size:12px">История пуста</span>';
    return;
  }
  h.forEach(item=>{
    const el = document.createElement('div');
    el.className = 'hchip';
    el.title = item;
    el.textContent = item.length>46 ? item.slice(0,46)+'…' : item;
    el.onclick = ()=>{ qEl.value=item; qEl.focus(); toast('Вставлено из истории'); };
    histList.appendChild(el);
  });
}
function toggleHistory(){
  const open = histBox.style.display === 'block';
  histBox.style.display = open ? 'none' : 'block';
  if(!open) renderHistory();
}
function clearHistory(){
  saveHistory([]);
  renderHistory();
  toast('История очищена');
}

function clearAll(){
  qEl.value = '';
  outEl.innerHTML = '<div class="empty">Введите симптомы и нажмите «Диагностировать».</div>';
  setError('');
  latEl.textContent = '—';
  dbg.style.display = 'none';
  rawEl.textContent = '';
  tbl.innerHTML = '';
  kv.innerHTML = '';
  LAST = null;
}

async function go(){
  const q = qEl.value.trim();
  if(!q){ setError('Введите симптомы пациента.'); qEl.focus(); return; }
  setError('');
  setLoading(true);
  outEl.innerHTML = '<div class="empty">⏳ Анализирую… (коды → HyDE → протоколы → диагноз)</div>';
  latEl.textContent = '…';
  dbg.style.display = 'none';

  const t0 = performance.now();
  timerEl.textContent = '0.0s';
  tick = setInterval(()=>{
    const t = (performance.now()-t0)/1000;
    timerEl.textContent = t.toFixed(1)+'s';
  }, 100);

  try{
    const r = await fetch('/diagnose_debug', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symptoms:q})
    });
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d = await r.json();
    LAST = d;

    const total = ((performance.now()-t0)/1000).toFixed(2);
    latEl.textContent = '⏱ ' + total + 's';

    pushHistory(q);
    renderResult(d);
    renderDebug(d);
  }catch(e){
    setError('Ошибка: ' + (e?.message || e));
    outEl.innerHTML = '';
    latEl.textContent = '—';
  }finally{
    clearInterval(tick); tick = null;
    setLoading(false);
  }
}

qEl.addEventListener('keydown', e=>{ if(e.ctrlKey && e.key==='Enter') go(); });
renderHistory();
</script>
</body>
</html>""",
        headers={"Cache-Control": "no-store"},
    )