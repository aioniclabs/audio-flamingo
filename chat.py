#!/usr/bin/env python3
"""
Audio Flamingo 3 – Terminal Chat
-----------------------------------------------------------------------
Usage (inside container, or after installing deps locally):
    python chat.py /path/to/audio.mp3
    python chat.py /path/to/audio.mp3 --think          # reasoning mode
    python chat.py /path/to/audio.mp3 --model nvidia/audio-flamingo-3-chat
-----------------------------------------------------------------------
The audio file is loaded once; every question you type refers to it.
Conversation history is passed back on each turn so the model can
follow up on previous answers.
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, think_mode: bool):
    import torch
    import llava
    from llava import conversation as clib
    from huggingface_hub import snapshot_download

    print(f"[*] Downloading / loading '{model_name}' …")
    model_path = snapshot_download(model_name)

    model = llava.load(model_path, device_map=None)

    if think_mode:
        from peft import PeftModel
        stage35_path = os.path.join(model_path, "stage35")
        if not os.path.isdir(stage35_path):
            print("[!] stage35 (think mode) not found in checkpoint – continuing without it.")
        else:
            nlt_path = os.path.join(stage35_path, "non_lora_trainables.bin")
            if os.path.exists(nlt_path):
                nlt = torch.load(nlt_path, map_location="cpu")
                nlt = {(k[6:] if k.startswith("model.") else k): v for k, v in nlt.items()}
                model.load_state_dict(nlt, strict=False)
            model = PeftModel.from_pretrained(
                model, stage35_path, device_map="auto", torch_dtype=torch.float16
            )
            print("[*] Think mode (Stage 3.5 LoRA) loaded.")

    model = model.to("cuda")

    # Use the 'auto' template – the model will self-identify the right format
    clib.default_conversation = clib.conv_templates["auto"].copy()
    print("[*] Model ready.\n")
    return model


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat(audio_path: str, model_name: str, think_mode: bool) -> None:
    from llava.media import Sound

    model = load_model(model_name, think_mode)
    sound = Sound(audio_path)

    print(f"Audio : {audio_path}")
    print("Type your question and press Enter.  'quit' / Ctrl-C to exit.")
    print("─" * 64)

    history: list[tuple[str, str]] = []   # [(user_msg, assistant_msg), …]

    while True:
        # ── prompt ──────────────────────────────────────────────────────────
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("[bye]")
            break

        # ── build text (with running history) ───────────────────────────────
        # AF3's generate_content is stateless, so we reconstruct the full
        # context on every call.  The audio object is always the first element.
        if history:
            prior = "\n".join(
                f"User: {u}\nAssistant: {a}" for u, a in history
            )
            text = f"Previous conversation:\n{prior}\n\nUser: {user_input}"
        else:
            text = user_input

        # ── inference ───────────────────────────────────────────────────────
        try:
            response = model.generate_content([sound, text])
        except Exception as exc:
            print(f"[error] {exc}")
            continue

        history.append((user_input, str(response)))

        print(f"\nAF3: {response}")
        print("─" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chat with Audio Flamingo 3 about an audio file"
    )
    parser.add_argument(
        "audio",
        help="Path to the audio file (.mp3 / .wav / .flac)",
    )
    parser.add_argument(
        "--model",
        default="nvidia/audio-flamingo-3",
        help="HuggingFace model ID  (default: nvidia/audio-flamingo-3)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Load Stage 3.5 LoRA for chain-of-thought reasoning",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[error] File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    ext = os.path.splitext(args.audio)[1].lower()
    if ext not in {".mp3", ".wav", ".flac"}:
        print(f"[warning] Unsupported extension '{ext}' – trying anyway.")

    chat(args.audio, args.model, args.think)


if __name__ == "__main__":
    main()
