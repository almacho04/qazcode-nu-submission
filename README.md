Below is **plain text only** (copy/paste) for your `README.md`. I adapted it to your submission (port **8080**, Docker run, UI, endpoints, env vars, LFS note, and “no external calls” note).

````md
# Datasaur 2026 | Qazcode Challenge

## Medical Diagnosis Assistant: Symptoms → ICD-10

An AI-powered clinical decision support system that converts patient symptoms into structured diagnoses with ICD-10 codes, grounded in Kazakhstan clinical protocols.

> ⚠️ Decision support only — not a substitute for clinical judgment.

---

## Challenge Overview

Users input symptoms as free text and receive:

- **Top-3 probable diagnoses** ranked by likelihood
- **ICD-10 codes** for each diagnosis
- **Brief clinical explanations** based on official Kazakhstan protocols

---

## Data Sources

### Kazakhstan Clinical Protocols
Official clinical guidelines serving as the primary knowledge base for diagnoses and diagnostic criteria.

Data format example:

```json
{"protocol_id":"p_d57148b2d4","source_file":"HELLP-СИНДРОМ.pdf","title":"Одобрен","icd_codes":["O00","O99"],"text":"..."}
````

---

## System Pipeline (Implementation)

This project uses a 3-stage pipeline:

1. **Stage 1 — Candidate ICD prediction**

   * Predicts candidate ICD-10 codes from symptoms (anchors retrieval).

2. **Stage 2 — Hybrid retrieval + reranking**

   * Retrieves protocol chunks from Kazakhstan protocols using:

     * ICD-anchored lookup
     * Semantic retrieval (dense embeddings + BM25 fusion)
     * Cross-encoder reranker for best evidence

3. **Stage 3 — Final diagnosis generation**

   * Generates **Top-3** diagnoses using ICD codes present in retrieved protocols.

Debug endpoint provides transparency: stage1 codes, retrieval evidence, and timings.

---

## Running Locally (Docker)

### 1) Build

From the repository root:

```bash
docker build -t submission .
```

### 2) Run (must serve on port 8080)

Recommended: use a local `.env` file (DO NOT COMMIT).

```bash
docker run --rm -p 8080:8080 --env-file .env submission
```

Open UI:

* [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

---

## API Endpoints

### POST `/diagnose`

Returns final Top-3 diagnoses JSON.

Example:

```bash
curl -s -X POST http://127.0.0.1:8080/diagnose \
  -H "Content-Type: application/json" \
  -d '{"symptoms":"Мужчина 55 лет. Давящая боль за грудиной 20 минут, одышка, холодный пот, иррадиация в левую руку."}' | jq
```

### POST `/diagnose_debug`

Returns full debug payload:

* stage1 candidate codes
* retrieval evidence (protocols + rerank scores)
* timings per stage
* final result JSON

```bash
curl -s -X POST "http://127.0.0.1:8080/diagnose_debug?top_k=10" \
  -H "Content-Type: application/json" \
  -d '{"symptoms":"Боль в животе, кровь в стуле, температура"}' | jq
```

---

## Environment Variables

This app supports OpenAI-compatible endpoints.

Required (if using external endpoint):

* `LLM_HUB_URL` — base URL (example: `https://hub.qazcode.ai/v1`)
* `LLM_API_KEY` — API key
* `LLM_MODEL` — model name (example: `openai/gpt-oss-120b`)

Optional:

* `HF_TOKEN` — improves HuggingFace download rate limits for embedding/reranker models

Example `.env` (DO NOT COMMIT):

```bash
LLM_HUB_URL=https://hub.qazcode.ai/v1
LLM_API_KEY=YOUR_KEY
LLM_MODEL=openai/gpt-oss-120b
HF_TOKEN=hf_xxx
```

---

## Evaluation

### Metrics

* **Accuracy@1**, **Recall@3**, **Latency**
* **Test set**: `data/test_set` (uses `query` and `gt` fields)
* **Holdout set**: private (not in repo)

### Run Evaluation

```bash
uv run python evaluate.py \
  -e http://127.0.0.1:8080/diagnose \
  -d ./data/test_set \
  -n <your_team_name> \
  -l 30 -p 1
```

Outputs saved to:

* `data/evals/<name>.jsonl`
* `data/evals/<name>_metrics.json`

---

## Large Files (Git LFS)

The retrieval index is large and tracked with Git LFS:

* `data/index/bm25.pkl`
* `data/index/chunks.json`
* `data/index/qdrant/**`

---

## Submission Checklist

* [ ] Everything packed into a single project (application, models, vector DB, indexes)
* [ ] Image builds successfully: `docker build -t submission .`
* [ ] Container starts and serves on port 8080: `docker run -p 8080:8080 submission`
* [ ] Web UI accepts free-text symptoms input
* [ ] Endpoint accepts free-text symptoms (`POST /diagnose`)
* [ ] Returns top diagnoses with ICD-10 codes
* [ ] README with build and run instructions
* [ ] `.env` and secrets are NOT committed

---

## Repo Structure

* `src/` — FastAPI server + retrieval/generation pipeline
* `data/index/` — BM25 + chunks + Qdrant local store (Git LFS)
* `data/test_set/` — public evaluation set
* `evaluate.py` — evaluation runner
* `pyproject.toml` / `uv.lock` — dependencies
* `Dockerfile` — container build and run

```

If you want, tell me your **GitHub username + repo name** (not a key/token), and I’ll tailor the clone instructions and links exactly to your final submission repo.
::contentReference[oaicite:0]{index=0}
```
