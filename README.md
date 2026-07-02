# Agentic Customer-Support Assistant (RAG + Kubernetes)

A production-oriented **agentic** customer-support assistant over a fixed 10-entry
knowledge base. It retrieves answers with **semantic search**, then makes a real
agentic decision — **answer**, ask a **clarifying** question, or **decline**
out-of-scope requests — and runs **entirely on a single-node Kubernetes cluster**
(FastAPI + bge-small embeddings + an in-memory vector index + Ollama serving
`qwen2.5:0.5b`).

> **Design thesis.** With only 10 entries, *retrieval is trivial* (recall@1 = 100%).
> The real engineering problem — and what this project optimises for — is a
> **principled, calibrated, observable decision policy** with **safe abstention**.
> Every threshold and tool choice below is derived from data and is meant to be
> defended, not hand-waved.

---

## 1. What it does

```
$ curl -s :8080/chat -d '{"question":"I forgot my login, how do I get back in?"}'
{
  "decision": "answer",
  "answer": "Go to the login page and click \"Forgot Password.\" … link expires after 30 minutes …",
  "sources": [{"id": "Q1", "category": "Account", "score": 0.7066}, ...],
  "scores": {"top1": 0.7066, "top2": 0.5921, "margin": 0.1145},
  "latency_ms": 1619,
  "model": "qwen2.5:0.5b"
}
```

Every response carries a **decision trace** (the chosen branch, the retrieved
sources, and the similarity scores) so the agentic behaviour is *observable*, not a
black box.

| Branch | When | Example |
|---|---|---|
| **answer** | confident single hit | "How do I reset my password?" → grounded answer, cites **Q1** |
| **clarify** | in-domain but ambiguous / near-tie | "I want to cancel and get my money back." → asks Billing-vs-refund |
| **decline** | out-of-scope / low similarity | "What's the weather today?" or "write code for me" → polite refusal |

---

## 2. Architecture

```mermaid
flowchart TD
    C[Client: curl / chat UI] -->|POST /chat| API
    subgraph Pod: support-agent (FastAPI, 2 replicas)
      API[Validate + correlation-id] --> E[Embed query<br/>bge-small + query prefix]
      E --> R[Retrieve top-k<br/>in-memory cosine]
      R --> RT{ROUTE<br/>s1, margin = s1 - s2}
      RT -->|s1 &lt; t_low OR top hit = Q10| DEC[DECLINE]
      RT -->|t_low ≤ s1 &lt; t_high<br/>OR margin &lt; t_margin| CLR[CLARIFY]
      RT -->|s1 ≥ t_high AND margin ≥ t_margin| ANS[ANSWER]
      ANS --> GEN[Grounded generation<br/>temp 0 + scope guard]
      GEN -->|guard says OUT_OF_SCOPE| DEC
      GEN -.Ollama down.-> EXT[Extractive fallback:<br/>return KB answer]
    end
    GEN -->|httpx| OLL[(Ollama<br/>qwen2.5:0.5b<br/>PVC + initContainer pull)]
    R --- IDX[(Baked KB index<br/>10 × 384 vectors)]
```

**Two times, don't conflate them.** *Ingest time* (once, at image build): embed the
10 entries → bake `data/index.npz`. *Query time* (per request): embed the query →
retrieve → route → (maybe) generate.

**The decision is made in deterministic Python, before the LLM is called.** The LLM
only *phrases* an answer and acts as a secondary grounding/scope guard that can only
ever *downgrade* answer → decline (fail-safe). This keeps the policy testable and
keeps a weak 0.5B model from driving control flow.

---

## 3. The agentic decision policy (the heart)

Two signals from the L2-normalised cosine similarities over the 10 single-chunk entries:

- `s1` = top-1 similarity
- `margin` = `s1 − s2` (how clearly the winner beats the runner-up)

```
DECLINE  if s1 < t_low                         # nothing close → out of scope
DECLINE  if top hit is the "Out of Scope" exemplar (Q10)
ANSWER   if s1 ≥ t_high AND margin ≥ t_margin   # confident, clear winner
CLARIFY  otherwise                              # weak (t_low ≤ s1 < t_high) or near-tie
```

**Calibrated, not guessed.** Thresholds come from a grid-search over the labelled
golden set (`make eval`), maximising routing accuracy and breaking ties by *fewest
"decline-routed-as-answer"* errors — the costliest mistake, since a confidently
wrong answer is worse than an unnecessary clarification.

| Threshold | Value | Meaning |
|---|---|---|
| `t_high` | **0.59** | min top-1 similarity to answer |
| `t_low` | **0.56** | below this ⇒ decline (out of scope) |
| `t_margin` | **0.02** | min top-1/top-2 gap to answer vs clarify |

**Out-of-scope is never string-matched against Q10.** It is derived from (1) low
absolute similarity and (2) Q10 acting as a semantic *exemplar* (coding requests land
near it) — both **deterministic**. A brand-new out-of-scope question therefore still
declines. On top, (3) the answer path runs a **best-effort** LLM scope guard for the
adversarial **Q7-vs-Q10** case ("write code to call your API" resembles the
integrations entry). Honest caveat: at 0.5B this guard is unreliable — it catches some
phrasings (*"write a Python script…"*) and misses others (*"write a SQL query…"*), so a
code request that lands in the ambiguous band **degrades to a clarifying question, never
a wrong answer**. A larger guard model (or a learned intent classifier — see Future work)
makes it reliable; the deterministic layers (1)+(2) are the primary defense.

---

## 4. Quickstart (single-node Kubernetes)

**Prereqs:** Docker, [kind](https://kind.sigs.k8s.io/), `kubectl`, `make`.

```bash
make deploy        # create kind cluster + build/load image + apply manifests + wait
make pf            # (separate terminal) port-forward svc to http://127.0.0.1:8080
make demo          # exercise answer / clarify / decline against the live cluster
make down          # tear everything down
```

`make deploy` is fully reproducible from a clean machine: Ollama pulls the model
into a PVC via an initContainer, the KB index is baked into the agent image, and
nothing is seeded by hand.

**Run locally without Kubernetes** — one command via Docker Compose (brings up
Ollama, pulls the model, starts the agent):

```bash
make compose-up            # or: docker compose up --build
# → open http://localhost:8000  (chat UI)
```

Or run just the API against a host Ollama:

```bash
pip install -r requirements-dev.txt
ollama pull qwen2.5:0.5b && ollama serve &
make run           # uvicorn on http://127.0.0.1:8000  (open / for the chat UI)
```

---

## 5. API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | `{"question": "..."}` → decision + answer + trace |
| `GET` | `/healthz` | liveness (process only — never checks deps) |
| `GET` | `/readyz` | readiness (gates traffic on Ollama reachability) |
| `GET` | `/metrics` | Prometheus text: request count, decision distribution, latency |
| `GET` | `/docs` | OpenAPI / Swagger UI |
| `GET` | `/` | minimal chat UI |

Out-of-scope example (note the decision comes from a **low score**, not a keyword):

```jsonc
// POST /chat {"question": "What's the weather in Essen today?"}
{ "decision": "decline", "answer": "That's outside the scope of this assistant. …",
  "scores": {"top1": 0.3976, "top2": 0.3773, "margin": 0.0203}, ... }
```

---

## 6. Design decisions & trade-offs

| Component | Choice | Why | Rejected alternative |
|---|---|---|---|
| **LLM serving** | Ollama + `qwen2.5:0.5b` | mandated; clean REST; 32K native context; ~400 MB, CPU-friendly | TinyLlama (2048-ctx cap); vLLM/TGI (GPU throughput servers — overkill) |
| **Embeddings** | `bge-small-en-v1.5` via **fastembed/ONNX** | 384-dim, 512-token, CPU-fast; ONNX drops the torch+CUDA tree (~0.5–2 GB) | sentence-transformers + torch (huge image, GPU deps we never use) |
| **Vector store** | **in-memory exact cosine** | N=10 ⇒ brute force is *exact*, sub-ms, zero ops | Qdrant/Chroma/pgvector (ANN over 10 vectors is over-engineering — see §8) |
| **Orchestration** | plain Python state machine | the decision is the headline; a 3-way code gate is testable & deterministic | LangGraph (loops/memory we don't need at this scope) |
| **Out-of-scope** | low-similarity + Q10 exemplar + LLM guard | generalises to unseen queries; no Q10 string-match | hardcoding Q10's text (won't generalise — a red flag) |
| **Manifests** | raw YAML + Kustomize base | "clean declarative manifests" is the literal ask; `configMapGenerator` auto-rolls pods on config change | Helm (templating indirection for one app) |
| **Cluster** | kind (one node) | reproducible; `kind load` removes the registry dependency | minikube/k3s — fine, but pick one and make it bulletproof |

**The two opinions worth defending:** *fastembed over torch* ("we never touch a
GPU") and *in-memory over a vector DB* ("ANN for 10 rows is a smell; here's exactly
when I'd switch to Qdrant").

---

## 7. RAG quality & evaluation

`make eval` runs the **real** embedding model over a labelled golden set
(`eval/golden.yaml`, 35 queries) and writes [`eval/results.md`](eval/results.md).
Latest run:

| Metric | Value |
|---|---|
| Retrieval **recall@1** | **100%** (21 answerable paraphrases) |
| Retrieval **MRR** | **1.000** |
| Routing accuracy (deterministic router, offline) | **82.9%** |
| Answer-branch recall | **100%** |

**Honest error analysis** (also in `results.md`): the residual routing misses are
two *inherent* limits of similarity-only routing on a tiny corpus — (1) genuinely
ambiguous queries for which bge-small still finds a confident nearest neighbour
(margin can't flag them), and (2) off-topic queries that share vocabulary with the
KB and clear `t_low`. Both are exactly what the **LLM scope guard** and the Q10
exemplar exist to catch at runtime. Reporting the router *in isolation* keeps the
calibration story honest.

---

## 8. Scaling to production (what changes past N=10 / one node)

The retriever lives behind a `Retriever` protocol, so the swap is a single file.

- **Vector store:** move from in-memory to **Qdrant** (HNSW index, payload
  filtering, persistence, horizontal sharding). In-memory brute force is `O(N·d)`
  per query — fine to ~10⁴–10⁵ vectors; beyond that, ANN (HNSW: tune `m`,
  `ef_construct`, `ef_search` for the recall/latency trade-off) becomes necessary.
  `pgvector` if you want vectors next to relational data; managed ANN if you don't
  want to run it.
- **Ingestion** becomes a real pipeline: a Kubernetes **Job** that chunks, embeds,
  and **upserts-by-id** into Qdrant (idempotent, no delete-then-insert window),
  triggered on KB change — instead of baking a static index.
- **Routing:** calibrate thresholds per-deployment from logged score distributions;
  add a learned out-of-scope/intent classifier (or an LLM judge with caching) to
  replace the heuristic guard; add a reranker + hybrid (BM25 + dense) search as the
  KB grows and lexical edge-cases appear.
- **LLM:** larger model on GPU via vLLM/TGI with continuous batching; cache frequent
  answers; stream tokens.
- **Ops:** HPA on the stateless agent, Ollama as a separate scaled service, proper
  Prometheus/Grafana + tracing, network policies, and secrets via a real manager.

---

## 9. Code quality, testing & observability

- **Modular** packages (`config`, `models`, `embeddings`, `retriever`, `agent`,
  `llm_client`, `api`, `metrics`); full type hints; **no magic numbers** (all
  tunables in `pydantic-settings` / the ConfigMap).
- **Tests:** `pytest` — router branch boundaries on synthetic scores, HTTP contract
  tests with a mocked LLM (fast, offline), and a real-embedding retrieval test
  (hit@1 on paraphrases). `ruff` + `mypy --strict`-ish + a docker build/offline
  smoke test run in **CI** (`.github/workflows/ci.yml`).
- **Resilience:** Ollama down → extractive fallback (return the KB answer verbatim,
  never a 500); empty/low retrieval → decline.
- **Observability:** structured JSON logs with a per-request correlation id; the
  decision trace in every response; `/metrics` exposing the decision distribution.
- **Security:** non-root container, read-only root FS, dropped capabilities, no
  secrets in code, input validation, model pulled offline at build.

---

## 10. Project structure

```
app/         config · models · embeddings · retriever · agent (router) · llm_client · api · metrics · logging
ingest/      idempotent index builder (baked at build; Job at scale)
eval/        golden.yaml · run_eval.py (calibration) · results.md
tests/       router · api contract · retrieval
k8s/         namespace · config.env · ollama · agent · kustomization
static/      minimal chat UI
Dockerfile   multi-stage; bakes model + index; non-root; offline runtime
Makefile     deploy · pf · demo · eval · test · lint · down
```

---

## 11. Limitations & future work

- **Ambiguity detection is similarity-only** today; a learned intent/ambiguity head
  or a (cached) LLM judge would lift clarify recall.
- **No multi-turn memory** — the clarify branch asks a question but doesn't yet
  consume the follow-up in a session. A short LangGraph loop is the natural next step.
- **No reranker / hybrid search** — unnecessary at N=10, listed in §8 for scale.
- **Thresholds are calibrated on a small hand-labelled set**; production should
  recalibrate from real traffic.

---

## 12. Notes for reviewers

- Git history is intentionally incremental: scaffold → retrieval → agent/router →
  API → Docker → k8s → eval → CI → docs.
- No secrets, model weights, or virtualenvs are committed; the embedding model is
  pulled at image-build time and baked into the runtime layer.

_MIT-licensed. Built for the IKIM agentic-RAG coding challenge._
