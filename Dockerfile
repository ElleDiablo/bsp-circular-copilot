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

# 5. Now the application code.
COPY app ./app

# 6. Don't run as root: if someone breaks into the app, they get a powerless user.
RUN useradd --create-home appuser
USER appuser

# 7. Documentation: the app listens on 8080 (App Runner's default port).
EXPOSE 8080

# 8. What runs when a container starts. --host 0.0.0.0 = accept connections from outside the
#    container (127.0.0.1 would only accept from inside it, so your Mac couldn't reach it).
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]