# ──────────────────────────────────────────────────────────────
# rocm-serve: Production LLM inference server for AMD ROCm
#
# Multi-stage build:
#   Stage 1: Builder — install dependencies and compile HIP kernels
#   Stage 2: Runtime — minimal image with only serving components
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM rocm/rocm-dev:6.1.2-ubuntu-22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=6.1.2
ARG PYTHON_VERSION=3.11

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Build custom HIP kernels
COPY CMakeLists.txt .
COPY src/kernels/ src/kernels/
COPY include/ include/
RUN mkdir build && cd build \
    && cmake -DCMAKE_PREFIX_PATH=/opt/rocm .. \
    && make -j$(nproc) \
    && make install

# Install rocm-serve
COPY . .
RUN pip install --no-cache-dir -e .

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM rocm/rocm-dev:6.1.2-ubuntu-22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/dist-packages/ \
    /usr/local/lib/python${PYTHON_VERSION}/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy HIP kernel libraries
COPY --from=builder /usr/local/lib/librocm_kernels.a /usr/local/lib/
COPY --from=builder /usr/local/include/rocm-serve/ /usr/local/include/rocm-serve/

# Copy application source
WORKDIR /app
COPY src/ ./src/
COPY LICENSE .

# Create non-root user
RUN useradd --create-home --shell /bin/bash rocm-serve
USER rocm-serve

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["python", "-m", "src.server"]
CMD ["--model", "meta-llama/Llama-3-8B-Instruct", "--port", "8000"]
