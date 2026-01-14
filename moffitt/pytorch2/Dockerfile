
## Builder stage to create artifacts
# Install uv
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

# Enable bytecode compilation, ensure platform-independent
ENV UV_COMPILE_BYTECODE=1

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable

# Copy the project into the intermediate image
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

## Runtime stage to executes artifacts
# Installed 3rd-party packages, your project package, console entrypoints and metadata are copied into the runtime image, but uv, build caches, lockfiles source tree and system packages installed in builder stage are excluded.
FROM python:3.13-slim

# Copy the environment, but not the source code, --chown=user:group src dst
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Create directories for data and exports if they don't exist
RUN mkdir -p data model_export forecasts plots

# Default command (can be overridden in docker-compose)
CMD ["python", "train_suite.py"]

