# Cloud deployment

The same code runs in two places. Locally it talks to the three containers in
`docker-compose.yml`; deployed it talks to managed equivalents. Nothing is
branched on a `PRODUCTION` flag — the difference is four environment variables,
which is the only version of "it works on my machine" that is actually testable,
because you can set those four variables locally and run the deployed
configuration on your laptop.

```
                    local                        deployed
  UI          streamlit run app/main.py    Streamlit Community Cloud (free)
  vectors     qdrant container :6333       Qdrant Cloud, 1 GB free cluster
  query log   postgres container :5432     Neon serverless Postgres (free)
  dashboard   grafana container :3000      Grafana container, pointed at Neon
```

## Why Grafana stays local

Everything else moves; the dashboard does not, and that is a decision rather
than an omission. Grafana is an operator's tool, not a visitor's. Hosting it
would mean either exposing an anonymous-viewer instance of the query log to the
public internet — every question anyone types, readable by anyone — or managing
a fourth set of credentials for an audience of one. Running it locally against
the production database costs one environment variable and reads exactly the
same rows:

```bash
POSTGRES_DSN='postgresql://...neon.tech/artmat?sslmode=require' docker compose up -d grafana
```

The dashboard is provisioned from `grafana/`, so it comes up already built
against whichever database it was pointed at.

## What you have to do yourself

Two accounts, both of which sign in with GitHub, and both free tiers with no
card required. Everything else is commands in this repo.

**1. Qdrant Cloud** — <https://cloud.qdrant.io>. Create a free 1 GB cluster;
pick a region near you (`eu-central-1` if you are in Europe — every query pays
this round trip). Copy the cluster URL and create an API key. The corpus is
4,115 points × 384 dimensions plus sparse vectors, roughly 40 MB, so the free
tier is not close to full.

**2. Neon** — <https://neon.tech>. Create a project, database name `artmat`.
Copy the **pooled** connection string, the one with `-pooler` in the hostname.
Streamlit re-runs the script on every widget interaction and each rerun opens a
connection; the direct endpoint runs out of them well before the pooler does.

Neon's free tier suspends the database after five minutes idle and wakes it on
the next connection. That wake is why `app/db.py` sets `connect_timeout=10` and
why the UI treats a logging failure as degraded monitoring rather than a failed
answer.

## Load the corpus into the cloud cluster

Local Qdrant stays as it is. Point the indexer at the cloud one for a single
run:

```bash
QDRANT_URL='https://<cluster>.aws.cloud.qdrant.io:6333' QDRANT_API_KEY='...' make index
```

Embedding runs on your machine — 4,115 chunks through ONNX on CPU, about 8
minutes, no API key and no cost — and only the vectors cross the network. Verify:

```bash
QDRANT_URL='https://<cluster>.aws.cloud.qdrant.io:6333' QDRANT_API_KEY='...' \
  .venv/bin/python -c "from rag.index import connect, COLLECTION; \
  print(connect().count(COLLECTION))"
```

Expect `count=4115`.

## Deploy the UI

Push this repo to GitHub, then at <https://share.streamlit.io>: **New app** →
this repository → branch `main` → main file `app/main.py` → **Advanced
settings** → Python 3.12 → paste the contents of `.streamlit/secrets.toml.example`
into the Secrets box with the real values filled in.

Python 3.12 rather than the default is deliberate: `onnxruntime==1.28.0` is
pinned in `requirements.txt`, and pinning a version that has no wheel for the
interpreter the platform happens to default to is how a reproducible build
becomes a ten-minute source compile that then fails.

The secrets box is the only place any credential is typed. `app/secrets.py`
copies the known names into `os.environ` with `setdefault`, so the rest of the
code keeps reading the environment and stays deployable anywhere else.

## First request is slow, once

A cold Streamlit container downloads the 130 MB `bge-small` ONNX model before it
can embed anything. `bootstrap()` pays that at boot by running one throwaway
search, so it lands on the deploy rather than on a visitor — but a container
that has been asleep will still take about a minute to wake. Subsequent queries
retrieve in ~30 ms and stream the first token in ~3 s.

The `Dockerfile` bakes both models into the image for anywhere that accepts a
container (Fly, Render, Hugging Face Spaces, a VM), which removes the cold
download entirely at the cost of image size.

## What it costs

Nothing but the Anthropic calls. Retrieval is free: embedding is local ONNX on
CPU, and both managed tiers are free. Each question is one Haiku 4.5 rewrite
plus one Sonnet 5 answer over ~8 passages — measured $0.01–0.03, and the exact
figure for every query is stored in `queries.cost_usd` and totalled on the
dashboard rather than estimated here.

The API key sits in one place, Streamlit's secrets store, and is not in the
repository. If it ever appears anywhere else — a screenshot, a paste, a log —
rotate it in the Anthropic console; the cost of doing that is a redeploy.
