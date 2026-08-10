# One command per pipeline stage, in dependency order. `make all` reproduces
# the system from an empty database; every target is independently re-runnable
# because each writes a file the next one reads.
PY := .venv/bin/python

.PHONY: help setup up down ingest chunk index eval-retrieval eval-judge ui all clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## create venv and install pinned dependencies
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

up:  ## start Qdrant, Postgres, Grafana
	docker compose up -d

down:  ## stop the stack (volumes preserved)
	docker compose down

ingest:  ## scrape and collect the three corpora (respects on-disk caches)
	$(PY) -m ingestion.sources.smoothon
	$(PY) -m ingestion.sources.openalex
	$(PY) -m ingestion.sources.tate
	$(PY) -m ingestion.relevance_gate --source-type materials_science

chunk:  ## boilerplate filter, spec/narrative split, abstract windowing
	$(PY) -m ingestion.chunk

index: up  ## embed and load 4,115 chunks into Qdrant (~7 min, CPU, no API key)
	$(PY) -m rag.index --drop

eval-retrieval:  ## 200 question pairs x 5 retrieval methods
	$(PY) -m eval.ground_truth
	$(PY) -m eval.retrieval

eval-judge:  ## 4 answer conditions x 25 questions, scored by Opus
	$(PY) -m eval.judge --n 25

ui: up  ## run the Streamlit interface on :8501
	$(PY) -m streamlit run app/main.py --server.port 8501

all: setup up ingest chunk index eval-retrieval eval-judge  ## full rebuild

clean:  ## drop containers and volumes -- destroys the index and query log
	docker compose down -v
