###############################################################################
# Stage 1 – build flash_attn
# Needs nvcc (devel image) to compile the CUDA extension.
###############################################################################
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04 AS flash_builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3 python3-pip python3-venv python3-dev \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch 2.6 built against CUDA 12.4 – fully compatible with CUDA 12.6 runtime
RUN pip install --upgrade pip && \
    pip install torch==2.6.0 packaging ninja \
        --index-url https://download.pytorch.org/whl/cu124

# Compile flash_attn and save the wheel so the runtime stage can install it
# without needing nvcc.
RUN pip wheel flash_attn==2.7.4.post1 --no-build-isolation -w /flash_wheels


###############################################################################
# Stage 2 – runtime image (what the user asked for)
###############################################################################
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        python3 python3-pip python3-venv \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# --- Python environment ------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Install the pre-built flash_attn wheel from the builder stage
COPY --from=flash_builder /flash_wheels /flash_wheels

# Same torch variant used when compiling flash_attn
RUN pip install torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
RUN pip install /flash_wheels/flash_attn*.whl

# --- Audio Flamingo 3 source -------------------------------------------------
RUN git clone --branch audio_flamingo_3 --depth 1 \
        https://github.com/NVIDIA/audio-flamingo /app

WORKDIR /app

# Install the llava package from the cloned source tree
RUN pip install -e .

# --- Inference dependencies --------------------------------------------------
# Keeping this minimal (no training/eval extras like deepspeed, wandb, gradio).
# If you hit a missing import, full deps are in docker/requirements.txt.
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
    openai-whisper==20240930 \
    pydub==0.25.1 \
    "av==14.2.0" \
    termcolor \
    "pydantic>=2.0" \
    "triton==3.1.0" \
    kaldiio \
    beartype \
    "bitsandbytes==0.43.2" \
    "decord==0.6.0" \
    "s2wrapper @ git+https://github.com/bfshi/scaling_on_scales@9c008a37540e761f53574b488979db6e49a64312"

# --- Chat script -------------------------------------------------------------
COPY chat.py /app/chat.py

# HuggingFace cache – mount a host directory here to avoid re-downloading
# the 7B model on every container run.
ENV HF_HOME=/data/hf_cache
VOLUME ["/data"]

ENTRYPOINT ["python3", "/app/chat.py"]
