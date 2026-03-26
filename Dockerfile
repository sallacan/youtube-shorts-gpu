FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py handler.py ./

# Runtime asset directories
RUN mkdir -p /workspace/music /workspace/fonts /workspace/outputs

CMD ["python3", "-u", "handler.py"]
