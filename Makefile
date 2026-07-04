IMAGE   ?= support-agent:0.1.0
CLUSTER ?= support
NS      ?= support-agent

.PHONY: help build cluster load deploy wait pf demo test lint fmt run down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

build: ## Build the agent image (bakes model + index)
	docker build -t $(IMAGE) .

cluster: ## Create a single-node kind cluster (idempotent)
	@kind get clusters | grep -qx $(CLUSTER) || kind create cluster --name $(CLUSTER)

load: build ## Load the image into kind (no registry needed)
	kind load docker-image $(IMAGE) --name $(CLUSTER)

deploy: cluster load ## One-command bring-up: cluster + image + manifests
	kubectl apply -k k8s
	@$(MAKE) wait

wait: ## Wait for Ollama (pulls model) + agent to be ready
	kubectl -n $(NS) rollout status deploy/ollama --timeout=600s
	kubectl -n $(NS) rollout status deploy/support-agent --timeout=300s

pf: ## Port-forward the agent to http://127.0.0.1:8080
	kubectl -n $(NS) port-forward svc/support-agent 8080:80

demo: ## Exercise all three branches (run `make pf` in another terminal first)
	./scripts/demo.sh

test: ## Run the unit + contract tests
	pytest

lint: ## ruff lint + format check + mypy
	ruff check app ingest tests
	ruff format --check app ingest tests
	mypy app ingest

fmt: ## Auto-format + autofix
	ruff format app ingest tests
	ruff check --fix app ingest tests

run: ## Run the API locally (needs Ollama at $$OLLAMA_BASE_URL)
	uvicorn app.api:app --reload

compose-up: ## Run the whole stack locally without Kubernetes (Ollama + model + agent)
	docker compose up --build

down: ## Delete the kind cluster
	kind delete cluster --name $(CLUSTER)

clean: ## Remove the local baked index
	rm -f data/index.npz
