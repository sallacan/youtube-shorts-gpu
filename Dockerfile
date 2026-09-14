# PyTorch 2.7.1 + CUDA 12.8 is the first stable line with kernels for Blackwell (sm_120).
# 2.4.1+cu121 stopped at sm_90, so any RTX PRO 6000 Blackwell MIG slice RunPod handed out
# crashed at CUDA init - and RunPod pools those slices with ordinary 24/48 GB cards even when
# gpuTypeIds lists exact model names. That forced the endpoints onto 80 GB H100s at ~8x the
# per-second price. With Blackwell supported, the cheap pool is usable again.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps (Python, PyTorch 2.7.1+cu128, torchvision, torchaudio already in base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy to avoid ABI issues with older compiled extensions
RUN pip install --no-cache-dir numpy==1.26.4

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake Kokoro TTS weights into the image. Without this, EVERY cold worker fetches
# them from HuggingFace at runtime; after an image rebuild all workers have an empty
# cache at once and HF answers 429 (this took down a production render on 2026-08-24).
ENV HF_HOME=/workspace/.cache/huggingface
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('hexgrad/Kokoro-82M')" \
    && du -sh /workspace/.cache/huggingface

# Copy app
COPY app.py handler.py ./

# Copy music files
COPY music/ /workspace/music/

# Runtime asset directories
RUN mkdir -p /workspace/fonts /workspace/outputs

CMD ["python3", "-u", "handler.py"]
