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

# Build image if it doesn't exist yet
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "[*] Building Docker image '$IMAGE' (first run – takes ~10-20 min) …"
    docker build -t "$IMAGE" "$(dirname "$0")"
fi

echo "[*] Starting chat for: $AUDIO_NAME"
docker run \
    --gpus all \
    --rm -it \
    -v "$AUDIO_DIR":/audio:ro \
    -v "${HF_CACHE_DIR:-$HOME/.cache/huggingface}":/data/hf_cache \
    "$IMAGE" \
    "/audio/$AUDIO_NAME" "${@:2}"
