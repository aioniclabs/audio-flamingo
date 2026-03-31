###############################################################################
# Single-stage build: use the devel image so flash_attn is compiled and run
# on the same system, eliminating ABI mismatches.
###############################################################################
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3 python3-pip python3-venv python3-dev \
        build-essential cmake pkg-config git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Python environment ------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# --- PyTorch -----------------------------------------------------------------
# cu124 wheels are compatible with the CUDA 12.6 runtime
RUN pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# --- flash_attn (compile in-place; nvcc is available in the devel image) -----
RUN pip install packaging ninja psutil numpy && \
    pip install flash_attn==2.7.4.post1 --no-build-isolation

# --- Audio Flamingo 3 source -------------------------------------------------
RUN git clone --branch audio_flamingo_3 --depth 1 \
        https://github.com/NVIDIA/audio-flamingo /app

WORKDIR /app

# Install the llava package from the cloned source tree
RUN pip install -e .

# --- Inference dependencies --------------------------------------------------
RUN pip install git+https://github.com/openai/whisper.git
RUN pip install \
    transformers==4.46.0 \
    tokenizers==0.20.3 \
    accelerate==0.34.2 \
    peft==0.15.1 \
    huggingface-hub==0.30.1 \
    safetensors==0.5.3 \
    einops==0.6.1 \
    "einops-exts==0.0.4" \
    timm==0.9.12 \
    librosa==0.11.0 \
    soundfile==0.13.1 \
    pydub==0.25.1 \
    "av==14.2.0" \
    termcolor \
    "pydantic>=2.0" \
    "triton==3.1.0" \
    kaldiio \
    beartype \
    "bitsandbytes==0.43.2" \
    "deepspeed==0.15.4" \
    "decord==0.6.0" \
    "s2wrapper @ git+https://github.com/bfshi/scaling_on_scales@9c008a37540e761f53574b488979db6e49a64312"

# Re-pin torch at the end to prevent any dependency from downgrading it
RUN pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 --no-deps

# --- API / server dependencies -----------------------------------------------
RUN pip install fastapi uvicorn[standard]

# --- Chat & server scripts ---------------------------------------------------
COPY chat.py /app/chat.py
COPY server.py /app/server.py

# HuggingFace cache – mount a host directory here to avoid re-downloading
# the 7B model on every container run.
ENV HF_HOME=/data/hf_cache
VOLUME ["/data"]

# Default: run the API server. Override CMD to use chat.py directly.
CMD ["python3", "/app/server.py"]
