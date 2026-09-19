# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.24@sha256:99ea34acedc870ba4ad11a1f540a1c04267c9f30aadc465a94406f52dfda2c36

FROM ${UV_IMAGE} AS uv

# Fetch pinned upstream sources so a clean clone builds without third_party/.
FROM ${PYTHON_IMAGE} AS sources
ARG OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
ARG LIBERO_COMMIT=f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
ENV GIT_LFS_SKIP_SMUDGE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git init /sources/openpi \
    && git -C /sources/openpi remote add origin https://github.com/Physical-Intelligence/openpi.git \
    && git -C /sources/openpi fetch --depth=1 origin ${OPENPI_COMMIT} \
    && git -C /sources/openpi checkout --detach FETCH_HEAD \
    && test "$(git -C /sources/openpi rev-parse HEAD)" = "${OPENPI_COMMIT}"

FROM sources AS libero-source
RUN git init /sources/libero \
    && git -C /sources/libero remote add origin https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    && git -C /sources/libero fetch --depth=1 origin ${LIBERO_COMMIT} \
    && git -C /sources/libero checkout --detach FETCH_HEAD \
    && test "$(git -C /sources/libero rev-parse HEAD)" = "${LIBERO_COMMIT}" \
    && rm -rf /sources/libero/.git

FROM ${PYTHON_IMAGE} AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && uv venv --python /usr/local/bin/python /opt/venv
COPY docker/requirements-cpu.lock /opt/locks/requirements-cpu.lock
RUN uv pip sync --python /opt/venv/bin/python --no-cache --require-hashes \
    --torch-backend cpu /opt/locks/requirements-cpu.lock
COPY --from=sources /sources/openpi/packages/openpi-client /opt/openpi-client
COPY --from=sources /sources/openpi/LICENSE /opt/licenses/openpi-LICENSE
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps --no-build-isolation /opt/openpi-client
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps --no-build-isolation . \
    && uv pip check --python /opt/venv/bin/python
COPY configs ./configs
COPY docker/smoke.py /opt/rl-vla/smoke.py
RUN mkdir -p /app/runs /app/checkpoints /app/datasets /home/app/.cache \
    && chown -R app:app /app/runs /app/checkpoints /app/datasets /home/app
USER app
ENTRYPOINT ["python", "-m", "rl_vla"]
CMD ["--help"]

FROM base AS test
USER root
COPY docker/requirements-test.lock /opt/locks/requirements-test.lock
RUN uv pip install --python /opt/venv/bin/python --no-cache --require-hashes \
    --torch-backend cpu -r /opt/locks/requirements-test.lock \
    && uv pip check --python /opt/venv/bin/python
COPY tests ./tests
# Windows build contexts can mark ordinary source files executable.
RUN find src tests -type f -name '*.py' -exec chmod 0644 '{}' +
USER app
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q", "-p", "no:cacheprovider"]

FROM base AS libero
USER root
# OSMesa works without an X server or GPU. EGL is available for native Linux GPU hosts.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libegl1 libglfw3 libglew2.2 libgles2 libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements-libero.lock /opt/locks/requirements-libero.lock
RUN apt-get update && apt-get install -y --no-install-recommends build-essential linux-libc-dev \
    && uv pip install --python /opt/venv/bin/python --no-cache --require-hashes \
        --torch-backend cpu -r /opt/locks/requirements-libero.lock \
    && apt-get purge -y --auto-remove build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=libero-source /sources/libero /opt/libero
# Upstream uses a namespace package that find_packages() misses. Compatibility
# editable mode puts the immutable source root on sys.path and preserves assets.
RUN uv pip install --python /opt/venv/bin/python --no-cache --no-deps --no-build-isolation \
        --config-setting editable_mode=compat -e /opt/libero \
    && uv pip check --python /opt/venv/bin/python
ENV LIBERO_CONFIG_PATH=/opt/libero-config \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    NUMBA_CACHE_DIR=/home/app/.cache/numba
COPY docker/libero-config.yaml /opt/libero-config/config.yaml
USER app
RUN python -c "from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv; assert 'libero_spatial' in benchmark.get_benchmark_dict()"

# Reproducible CUDA learner and lightweight benchmarks. OpenPI inference stays
# in its separate server image; no manual installs in a running container.
FROM libero AS gpu
USER root
COPY docker/requirements-gpu.lock /opt/locks/requirements-gpu.lock
RUN uv pip install --python /opt/venv/bin/python --no-cache --require-hashes \
        --torch-backend cu126 -r /opt/locks/requirements-gpu.lock \
    && uv pip check --python /opt/venv/bin/python
COPY docker/run_gpu_demos.py docker/benchmark_report.py /opt/rl-vla/
COPY tests ./tests
RUN find src tests -type f -name '*.py' -exec chmod 0644 '{}' +
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    SDL_VIDEODRIVER=dummy
USER app
# A build does not require a GPU. Actual CUDA access is checked by the runner.
RUN python -c "import torch, pygame, panda_gym, imageio, imageio_ffmpeg; assert torch.__version__ == '2.7.0+cu126'; assert torch.version.cuda == '12.6'"
ENTRYPOINT ["python", "/opt/rl-vla/run_gpu_demos.py"]
CMD ["--help"]

# A plain `docker build .` produces the small learner image, not the GPU server.
FROM base AS cpu
