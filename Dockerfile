FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps (Python 3.10, PyTorch 2.2.1+cu121 already in base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy to avoid ABI issues with older compiled extensions
RUN pip install --no-cache-dir numpy==1.26.4

# torchvision + torchaudio matching torch 2.2.1 in base
RUN pip install --no-cache-dir \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

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
