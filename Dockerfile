FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch==2.2.0 torchvision torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py handler.py ./

# Copy music files
COPY music/ /workspace/music/

# Runtime asset directories
RUN mkdir -p /workspace/fonts /workspace/outputs

CMD ["python3", "-u", "handler.py"]
