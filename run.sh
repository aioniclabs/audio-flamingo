#!/usr/bin/env bash
# Build and run the Audio Flamingo 3 chat container.
#
# Usage:
#   ./run.sh /path/to/audio.mp3          # normal mode
#   ./run.sh /path/to/audio.mp3 --think  # chain-of-thought reasoning mode
#
# The first run downloads the ~14 GB model into ~/.cache/huggingface.
# Subsequent runs reuse the cached weights via the -v mount.

set -euo pipefail

AUDIO_FILE="${1:?Usage: $0 <audio_file> [extra chat.py flags]}"
AUDIO_ABS="$(realpath "$AUDIO_FILE")"
AUDIO_DIR="$(dirname  "$AUDIO_ABS")"
AUDIO_NAME="$(basename "$AUDIO_ABS")"

IMAGE="audio-flamingo"

# Always rebuild so Dockerfile changes are picked up.
# Docker layer caching makes this fast when nothing changed.
echo "[*] Building Docker image '$IMAGE' …"
docker build -t "$IMAGE" "$(dirname "$0")"

echo "[*] Starting chat for: $AUDIO_NAME"
docker run \
    --gpus all \
    --rm -it \
    -v "$AUDIO_DIR":/audio:ro \
    -v "${HF_CACHE_DIR:-$HOME/.cache/huggingface}":/data/hf_cache \
    "$IMAGE" \
    "/audio/$AUDIO_NAME" "${@:2}"
