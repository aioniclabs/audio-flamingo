#!/usr/bin/env python3
"""
Audio Flamingo 3 – HTTP API Server
-----------------------------------------------------------------------
POST /chat
  Form fields:
    audio     – audio file upload (.mp3 / .wav / .flac)
    question  – text question about the audio
    history   – (optional) JSON array of [user, assistant] string pairs
                for multi-turn conversation

Response JSON:
    { "response": "<model answer>" }

Env vars:
    MODEL_NAME   HuggingFace model ID  (default: nvidia/audio-flamingo-3)
    THINK_MODE   Set to "1" to enable chain-of-thought LoRA
    PORT         Server port  (default: 5566)
-----------------------------------------------------------------------
"""

import json
import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from transformers import GenerationConfig

app = FastAPI(title="Audio Flamingo 3 API")

_model = None  # loaded once at startup


# ---------------------------------------------------------------------------
# Startup – load model
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    global _model
    from chat import load_model
    model_name = os.environ.get("MODEL_NAME", "nvidia/audio-flamingo-3")
    think_mode = os.environ.get("THINK_MODE", "").lower() in ("1", "true", "yes")
    _model = load_model(model_name, think_mode)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/chat")
async def chat_endpoint(
    audio: UploadFile = File(..., description="Audio file (.mp3/.wav/.flac)"),
    question: str = Form(..., description="Question about the audio"),
    history: str = Form(
        default="[]",
        description='JSON array of prior [user, assistant] string pairs, e.g. [["Q1","A1"],["Q2","A2"]]',
    ),
):
    from llava.media import Sound
    from chat import get_duration

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        sound = Sound(tmp_path)

        hist: list[list[str]] = json.loads(history)

        if hist:
            prior = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in hist)
            text = f"Previous conversation:\n{prior}\n\nUser: {question}"
        else:
            duration = get_duration(tmp_path)
            mins, secs = divmod(int(duration), 60)
            text = f"[This audio is {mins}:{secs:02d} long] {question}"

        gen_cfg = GenerationConfig(max_new_tokens=512, do_sample=True, temperature=0.7)
        response = _model.generate_content([sound, text], generation_config=gen_cfg)
        return JSONResponse({"response": str(response)})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5566))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
