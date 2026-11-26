# Dockerfile: the recipe for the image. Each line is one "layer"; Docker caches layers and
# rebuilds only from the first line whose input changed.

# 1. Start from a small official Linux image that already has Python 3.12.
FROM python:3.12-slim

# 2. Copy the uv binary out of uv's own image (no pip install needed). Pinned = repeatable builds.
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

# 3. All following commands run in /app inside the image.
WORKDIR /app

# 4. Dependencies first, code second: deps change rarely, code often. Copying only these two
#    files means editing app/*.py does NOT re-download every package (the layer stays cached).
COPY pyproject.toml uv.lock ./
#    --locked: fail if uv.lock is out of date; --no-dev: skip pytest/ruff;
#    --no-install-project: install the dependencies only (our code isn't copied yet).
RUN uv sync --locked --no-dev --no-install-project

# 5. The image's default settings = the cloud setup (no Ollama in the cloud). `docker run -e` overrides.
#    FASTEMBED_CACHE_PATH: where the embedding model is stored inside the image.
ENV EMBED_PROVIDER=fastembed LLM_PROVIDER=groq FASTEMBED_CACHE_PATH=/app/models

# 6. Don't run as root: if someone breaks into the app, they get a powerless user.
RUN useradd --create-home appuser

# 7. Bake the embedding model (~0.6 GB) into the image at BUILD time, so a starting container doesn't
#    download it. Before the code: editing app/*.py won't re-download it. chown = appuser owns it.
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding; TextEmbedding('mixedbread-ai/mxbai-embed-large-v1')" \
    && chown -R appuser /app/models
ENV HF_HUB_OFFLINE=1

# 8. Now the application code.
COPY app ./app
USER appuser

# 9. Documentation: the app listens on 8080 (App Runner's default port).
EXPOSE 8080

# 10. What runs when a container starts. --host 0.0.0.0 = accept connections from outside the
#    container (127.0.0.1 would only accept from inside it, so your Mac couldn't reach it).
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
