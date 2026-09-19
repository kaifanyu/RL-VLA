# syntax=docker/dockerfile:1.7
# Build from the repository root. The server deliberately has its own environment.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:63a18dd805367dacfb077aeced8384ab2fb569598ec5f5f5220c3f90a5c23650
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36
FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
ARG OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
ARG PYTHON_VERSION=3.11.13
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/openpi/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    GIT_LFS_SKIP_SMUDGE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential clang git git-lfs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/openpi
# Ignored local third_party checkouts and model weights are never build inputs.
RUN git init . \
    && git remote add origin https://github.com/Physical-Intelligence/openpi.git \
    && git fetch --depth=1 origin "${OPENPI_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${OPENPI_COMMIT}" \
    && printf '%s\n' "${OPENPI_COMMIT}" > /opt/openpi/OPENPI_COMMIT \
    && rm -rf /opt/openpi/.git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
    && uv sync --frozen --no-dev --no-editable --python "${PYTHON_VERSION}" \
    && uv pip check --python /opt/openpi/.venv/bin/python
# Upstream's required Transformers changes. Copy mode keeps the uv cache pristine.
RUN /opt/openpi/.venv/bin/python -c "import importlib.util, pathlib, shutil; source = pathlib.Path('src/openpi/models_pytorch/transformers_replace'); target = pathlib.Path(importlib.util.find_spec('transformers').origin).parent; shutil.copytree(source, target, dirs_exist_ok=True)"
# Import the real serving path on CPU at build time; runtime checks GPU access.
RUN JAX_PLATFORMS=cpu /opt/openpi/.venv/bin/python -c "import jax, torch, transformers; from openpi.policies import policy_config; from openpi.training import config; from openpi.serving import websocket_policy_server; from transformers.models.siglip.check import check_whether_transformers_replace_is_installed_correctly; assert jax.__version__ == '0.5.3'; assert torch.__version__.split('+')[0] == '2.7.1'; assert transformers.__version__ == '4.53.2'; assert check_whether_transformers_replace_is_installed_correctly(); config.get_config('pi05_libero')"

FROM base AS runtime
ARG OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
LABEL org.opencontainers.image.title="RL-VLA OpenPI policy server" \
      org.opencontainers.image.description="Isolated frozen OpenPI GPU inference service" \
      org.opencontainers.image.source="https://github.com/Physical-Intelligence/openpi" \
      org.opencontainers.image.revision="${OPENPI_COMMIT}"
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash app \
    && mkdir -p /checkpoints /home/app/.cache/openpi /home/app/.cache/huggingface /home/app/.cache/jax \
    && chown -R app:app /home/app /checkpoints
COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/openpi /opt/openpi
COPY docker/openpi_entrypoint.py docker/openpi_healthcheck.py /usr/local/bin/
ENV PATH="/opt/openpi/.venv/bin:${PATH}" \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache \
    OPENPI_DATA_HOME=/home/app/.cache/openpi \
    HF_HOME=/home/app/.cache/huggingface \
    JAX_COMPILATION_CACHE_DIR=/home/app/.cache/jax \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OPENPI_CONFIG=pi05_libero \
    OPENPI_CHECKPOINT=/checkpoints/pi05_libero \
    OPENPI_PORT=8000
WORKDIR /opt/openpi
USER app:app
EXPOSE 8000
# Readiness starts only after the checkpoint has loaded and the socket is serving.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD ["python", "/usr/local/bin/openpi_healthcheck.py"]
ENTRYPOINT ["python", "/usr/local/bin/openpi_entrypoint.py"]
CMD ["serve"]
