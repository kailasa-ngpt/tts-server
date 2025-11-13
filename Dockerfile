# Multi-stage Dockerfile providing CPU and GPU targets for the Unsloth CSM TTS server

# =============================
# GPU Target (CUDA 12.1 runtime)
# =============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential \
    libsndfile1 \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

COPY requirements.txt ./

# Torch (CUDA 12.1) and project dependencies
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio \
 && pip install triton \
 && pip install xformers \
 && pip install -r requirements.txt \
 && pip install --no-deps trl==0.22.2

# Copy source
COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app \
    TTS_BACKEND=unsloth_csm \
    HF_HOME=/root/.cache/huggingface

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================
# CPU Target
# =============================
FROM python:3.10-slim AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Torch (CPU) and project dependencies
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio \
 && pip install triton \
 && pip install -r requirements.txt \
 && pip install --no-deps trl==0.22.2

# Copy source
COPY . .

EXPOSE 8000

ENV PYTHONPATH=/app \
    TTS_BACKEND=unsloth_csm \
    HF_HOME=/root/.cache/huggingface

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
