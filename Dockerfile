# Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10 / SM 121 / ARM64 / CUDA 13)
#
# Builds an OpenAI-compatible TTS server with CUDA graph acceleration.
# Clones upstream faster-qwen3-tts at build time and applies DGX Spark fixes.

FROM nvidia/cuda:13.0.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and required audio/build tools
RUN apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3-pip python3.12-dev \
                       ffmpeg git curl sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the upstream faster-qwen3-tts library
ARG FASTER_QWEN3_TTS_REF=main
RUN git clone --depth 1 --branch ${FASTER_QWEN3_TTS_REF} \
    https://github.com/andimarafioti/faster-qwen3-tts.git /app

# Apply DGX Spark patches (non_streaming_mode=True, per-voice temperature/top_k/top_p)
COPY patches/openai_server.patch /tmp/
RUN cd /app && git apply /tmp/openai_server.patch

# Create virtual environment (Ubuntu 24.04 enforces PEP 668)
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

# Install ARM64 CUDA 13 wheels for PyTorch stack
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Install faster-qwen3-tts and server dependencies
RUN pip install --no-cache-dir -e ".[demo]"
RUN pip install --no-cache-dir pydub soundfile uvicorn fastapi

EXPOSE 8000

CMD ["python", "examples/openai_server.py", \
     "--model", "/models/Qwen3-TTS", \
     "--voices", "/config/voices.json", \
     "--port", "8000"]
