"""
OpenAI-compatible TTS server for Qwen3-TTS-12Hz-1.7B-VoiceDesign.

Voices are defined in voicedesign_voices.json as:
  { "voice_id": { "instruct": "...", "language": "..." } }

No ref_audio needed — the instruct text fully describes the voice.
"""
import json
import logging
import queue
import threading
import asyncio
import argparse
import numpy as np
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

sys.path.append("/app")
from faster_qwen3_tts.model import FasterQwen3TTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
tts_model: FasterQwen3TTS = None
voices: dict = {}
default_voice: str = None
SAMPLE_RATE = 24000
DEFAULT_MAX_NEW_TOKENS = 2048
_model_lock = threading.Lock()
_load_model_kwargs = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _do_load_and_warmup)
    yield


def _do_load_and_warmup():
    global tts_model, SAMPLE_RATE
    import torch
    args = _load_model_kwargs
    try:
        logger.info("Loading VoiceDesign model %s …", args.model)
        model = FasterQwen3TTS.from_pretrained(
            args.model,
            device=args.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            max_seq_len=args.max_seq_len,
        )
        SAMPLE_RATE = model.sample_rate
        logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)

        # Warmup
        logger.info("Warming up CUDA graphs (first request will be fast)...")
        try:
            for _ in model.generate_voice_design_streaming(
                text="Warmup.",
                instruct="Warmup.",
                language="English"
            ):
                pass
            logger.info("CUDA warmup complete — server ready.")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): %s", exc)

        tts_model = model
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request schema (OpenAI TTS compatible)
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "vd_british_male"
    response_format: str = "wav"
    speed: float = 1.0
    language: Optional[str] = None
    instruct: Optional[str] = None
    max_new_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _to_pcm16(audio: np.ndarray) -> bytes:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int) -> bytes:
    import struct
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", 0xFFFFFFFF,
    )


def _to_mp3_bytes(audio: np.ndarray, sr: int) -> bytes:
    from pydub import AudioSegment
    import io
    pcm = _to_pcm16(audio)
    seg = AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)
    buf = io.BytesIO()
    seg.export(buf, format="mp3")
    return buf.getvalue()


def resolve_voice(name: str) -> dict:
    cfg = voices.get(name)
    if cfg:
        return cfg
    if default_voice and default_voice in voices:
        logger.warning("Voice %r not found, falling back to %r", name, default_voice)
        return voices[default_voice]
    raise HTTPException(status_code=404, detail=f"Voice {name!r} not found")


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _request_generation_params(req: SpeechRequest, voice_cfg: dict) -> dict:
    instruct = req.instruct if req.instruct is not None else voice_cfg.get("instruct", "")
    language = req.language or voice_cfg.get("language", "English")
    return {
        "text": req.input,
        "instruct": instruct,
        "language": language,
        "max_new_tokens": req.max_new_tokens or int(voice_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
    }


async def _stream_chunks(params: dict):
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_design_streaming(**params):
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    threading.Thread(target=producer, daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice)
    params = _request_generation_params(req, voice_cfg)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg"}
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt!r}")

    if fmt == "mp3":
        loop = asyncio.get_event_loop()
        def _gen():
            with _model_lock:
                return tts_model.generate_voice_design(**params)
        audio_arrays, sr = await loop.run_in_executor(None, _gen)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        return Response(content=_to_mp3_bytes(audio, sr), media_type="audio/mpeg")

    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)
        async for raw in _stream_chunks(params):
            yield raw

    return StreamingResponse(audio_stream(), media_type=_CONTENT_TYPES[fmt])


_voice_list = None
_models_response = None


def _build_voice_list():
    global _voice_list, _models_response
    _voice_list = [{"id": v, "object": "model", "created": 1686935002, "owned_by": "qwen"} for v in voices]
    _models_response = {"object": "list", "data": _voice_list}


@app.get("/v1/models")
async def list_models():
    return _models_response

@app.get("/v1/audio/voices")
async def list_audio_voices():
    return _models_response

@app.get("/v1/audio/models")
async def list_audio_models():
    return _models_response

@app.get("/speakers")
async def get_speakers():
    return list(voices.keys())

@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global voices, default_voice, SAMPLE_RATE, DEFAULT_MAX_NEW_TOKENS, _load_model_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3-TTS-VoiceDesign")
    parser.add_argument("--voices", default="/config/voicedesign_voices.json")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    DEFAULT_MAX_NEW_TOKENS = args.max_seq_len
    _load_model_kwargs = args

    with open(args.voices) as f:
        voices = json.load(f)
    default_voice = next(iter(voices), None)
    _build_voice_list()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
