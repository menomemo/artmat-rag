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

A local `.streamlit/secrets.toml` **shadows `.env`**, and it does so silently.
`app/secrets.py` runs before `rag.env`, and `rag.env` loads `.env` with
`override=False` — so anything present in a local secrets file wins. That is
correct for the deployed app and a trap on a laptop: a secrets file left behind
after copying its contents into the console will keep pointing your local runs
at cloud services, with whatever key it held at the time. Copy it, paste it,
delete it.

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

## The same image on AWS (ECS Express Mode)

Streamlit Community Cloud installs `requirements.txt` itself and never touches
the `Dockerfile`. To show the image is the deployable unit rather than a
comment, the same container also ran on AWS from 2026-11-19, with no code
change: the four variables above, nothing else.

```
  UI          ECS Fargate via Express Mode — 1 vCPU / 2 GB, exactly 1 task,
              public HTTPS endpoint behind a managed ALB
  image       ECR, eu-west-2
  secrets     Secrets Manager, one secret per variable
  logs        CloudWatch (created by Express Mode)
  vectors     Qdrant Cloud, unchanged      query log   Neon, unchanged
```

Everything is in `eu-west-2`, the same region as the Qdrant cluster and the Neon
database, so no request crosses a region.

**Build for the target, not the laptop.** The development machine is Apple
Silicon; Fargate defaults to x86. An `arm64` image pushes fine and then fails
to start, so the platform is stated explicitly:

```bash
aws ecr create-repository --repository-name artmat-rag --region eu-west-2
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.eu-west-2.amazonaws.com
docker buildx build --platform linux/amd64 --provenance=false -t <account>.dkr.ecr.eu-west-2.amazonaws.com/artmat-rag:v1 --push .
```

`.dockerignore` already excludes `.env` and `.streamlit/secrets.toml`, so no
credential is baked into a layer.

**One secret per variable.** The obvious layout — one JSON secret, referenced
per key as `arn:...:secret:artmat-rag/prod-XXXX:QDRANT_URL::` — is rejected by
`create-express-gateway-service` with *Invalid resource in secrets manager ARN*.
Five plain-text secrets named `artmat-rag/<VARIABLE>` work, and the execution
role's policy is scoped to `secret:artmat-rag/*` rather than `*`. The container
receives them as environment variables, which is what `rag.env` reads.

**Two roles.** `ecsTaskExecutionRole` (managed
`AmazonECSTaskExecutionRolePolicy` plus `secretsmanager:GetSecretValue` on
`artmat-rag/*`) pulls the image and injects secrets.
`ecsInfrastructureRoleForExpressServices` (managed
`AmazonECSInfrastructureRoleforExpressGatewayServices`, trusted by
`ecs.amazonaws.com`) lets ECS create the load balancer, target group and
security groups.

**The service.** Express Mode defaults to auto-scaling between several tasks;
this app is a demo with one user at a time, so min and max are both 1:

```json
{
  "serviceName": "artmat-rag",
  "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
  "infrastructureRoleArn": "arn:aws:iam::<account>:role/ecsInfrastructureRoleForExpressServices",
  "healthCheckPath": "/_stcore/health",
  "primaryContainer": {
    "image": "<account>.dkr.ecr.eu-west-2.amazonaws.com/artmat-rag:v1",
    "containerPort": 8501,
    "secrets": [{"name": "QDRANT_URL", "valueFrom": "arn:aws:secretsmanager:eu-west-2:<account>:secret:artmat-rag/QDRANT_URL-XXXXXX"}]
  },
  "cpu": "1024",
  "memory": "2048",
  "scalingTarget": {"minTaskCount": 1, "maxTaskCount": 1, "autoScalingMetric": "AVERAGE_CPU", "autoScalingTargetValue": 70}
}
```

(`secrets` lists all five; one is shown.) Create it with
`aws ecs create-express-gateway-service --region eu-west-2 --cli-input-json file://service.json`.
Streamlit's own health endpoint is `/_stcore/health`; Express Mode's default
`/ping` does not exist here and would keep the target unhealthy forever.

Three things that look like failures and are not:

- **The endpoint does not resolve for the first few minutes.** A resolver that
  asked too early caches the negative answer; `dig` succeeds while `curl` still
  says *Could not resolve host*. `curl --resolve host:443:<ip>` proves the
  service is up without waiting for the cache.
- **Two tasks run after an update.** Express Mode starts the new revision before
  draining the old one; `runningCount` is 2 for a few minutes, then 1.
- **A 503 right after scaling up.** No healthy target yet; the model download
  is baked into the image, so it clears in about a minute.

**Verified** on 2026-11-19 from the public endpoint: a sample question was
rewritten, retrieved from Qdrant Cloud, answered with citations, and logged as
a new row in Neon (`queries.id = 3`, $0.031).

**Pause and teardown.** Setting min and max to 0 stops the Fargate charge but
keeps the load balancer (and its URL), which still bills by the hour. Deleting
the service removes the ALB; then delete the ECR repository and the
`artmat-rag/*` secrets. Roughly $2 a day while running, covered by credits;
the deployment is temporary by design and the Streamlit Cloud app is the
long-lived demo.

### The free Qdrant cluster is not a backup

On 2026-11-19 the original Qdrant Cloud cluster was gone: an idle free-tier
cluster is reclaimed, and both deployments failed with *connection reset by
peer*. `data/chunks.jsonl` is gitignored and was not on disk, so `make index`
could not rebuild it. The local compose volume still held the full collection,
which was moved across as a snapshot — no re-embedding:

```bash
curl -X POST "localhost:6333/collections/artmat/snapshots?wait=true"
curl -o artmat.snapshot "localhost:6333/collections/artmat/snapshots/<name>"
curl -X POST -H "api-key: $QDRANT_API_KEY" -F "snapshot=@artmat.snapshot" \
  "$QDRANT_URL/collections/artmat/snapshots/upload?priority=snapshot&wait=true"
```

The new cluster reported `green`, 4,115 points, `dense` + `bm25`. A 31 MB
snapshot is the cheapest insurance in the project; keep one somewhere that is
not a free tier.
