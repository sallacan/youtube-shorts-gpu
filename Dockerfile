FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps (Python, PyTorch 2.4.1+cu121, torchvision, torchaudio already in base)
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

# Copy app
COPY app.py handler.py ./

# Copy music files
COPY music/ /workspace/music/

# Runtime asset directories
RUN mkdir -p /workspace/fonts /workspace/outputs

CMD ["python3", "-u", "handler.py"]
