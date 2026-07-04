# Support RAG Agent

A small customer-support assistant over a fixed 10-entry knowledge base. It
retrieves the closest KB entry with semantic search, then decides whether to
**answer**, ask a **clarifying** question, or **decline** out-of-scope requests.
It ships with an admin panel (tickets, chat log, KB editor) and runs on FastAPI +
Ollama, locally or on a single-node Kubernetes cluster.

## Stack

- **API**: FastAPI
- **Embeddings**: `bge-small-en-v1.5` via fastembed (ONNX, CPU-only)
- **Retrieval**: in-memory cosine over the 10 entries (a vector DB is overkill at this size)
- **LLM**: Ollama running `qwen2.5:0.5b`
- **Storage**: SQLite (chat log + support tickets)

## How the agent decides

Retrieval gives a top-1 similarity (`s1`) and the gap to the runner-up
(`margin`). The route is plain Python, decided before the LLM is called:

- `s1 < t_low` or the top hit is the out-of-scope entry → **decline**
- `s1 >= t_high` and `margin >= t_margin` → **answer** (the LLM phrases the KB answer)
- otherwise → **clarify**

Thresholds live in `app/config.py` (or the ConfigMap) — `t_high=0.59`,
`t_low=0.56`, `t_margin=0.02`.

## Run it locally

Needs Ollama running with the model pulled:

```bash
pip install -r requirements-dev.txt
ollama pull qwen2.5:0.5b && ollama serve &
uvicorn app.api:app --reload      # http://127.0.0.1:8000
```

Ask it something:

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"question":"how do I reset my password?"}'
```

Or bring up the whole stack (Ollama + model + agent) with Docker:

```bash
docker compose up --build         # http://localhost:8000
```

## Run it on Kubernetes

Single-node with [kind](https://kind.sigs.k8s.io/):

```bash
make deploy      # create cluster + build/load image + apply k8s/
make pf          # port-forward to http://127.0.0.1:8080
make demo        # exercise answer / clarify / decline
make down        # tear it down
```

## Admin panel

`http://localhost:8000/admin`

- **Dashboard** – message and ticket counts
- **Chats** – every question logged with the visitor IP, the result, and score
- **Tickets** – opened automatically when the assistant declines; close or note them
- **Knowledge Base** – add / edit / delete entries; the index is rebuilt on save

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{"question": "..."}` → decision + answer + trace |
| GET | `/tickets` | list support tickets (JSON) |
| GET | `/healthz` | liveness |
| GET | `/readyz` | readiness (checks Ollama) |
| GET | `/metrics` | Prometheus counters |
| GET | `/docs` | Swagger UI |
| GET | `/admin` | admin panel |

## Layout

```
app/         config · models · embeddings · retriever · agent · llm_client · api
             storage (SQLite) · admin (panel) · kb (editor) · metrics · logging
ingest/      builds the retrieval index from the KB
templates/   admin panel pages
static/      chat UI
k8s/         namespace · config · ollama · agent · kustomization
tests/       router · retrieval · guard · API contract
```

## Tests

```bash
make test    # pytest
make lint    # ruff + mypy
```
