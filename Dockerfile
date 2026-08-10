# The app image. Streamlit Community Cloud does not use this -- it installs
# requirements.txt itself -- but a deployment that only works on one vendor's
# free tier is a demo, not a deployment. This runs the same app on Fly, Render,
# Hugging Face Spaces, or any VM, and it is what the compose stack would use to
# put the UI next to its own Qdrant and Postgres.
FROM python:3.12-slim

# fastembed pulls its ONNX models over HTTPS at first use and caches them here.
# Set explicitly so the cache lands somewhere writable by a non-root user
# rather than in a home directory the platform may not give us.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HOME=/opt/hf

WORKDIR /app

# Requirements first, so a code edit does not reinstall onnxruntime.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding models into the image. Downloading 130 MB on first request
# turns a cold start into a user who thinks the app is broken; downloading it
# at build time costs image size, which nobody is waiting on.
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25')"

COPY . .

RUN useradd --create-home artmat && chown -R artmat /opt/fastembed /opt/hf
USER artmat

EXPOSE 8501
# `--server.address=0.0.0.0` because Streamlit binds localhost by default,
# which inside a container means unreachable from outside it.
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
