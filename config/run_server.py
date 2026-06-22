"""
Wrapper around faster-qwen3-tts's openai_server.py that injects additional
API endpoints for compatibility with OpenWebUI and SillyTavern.

Endpoints added:
  GET /v1/models        - Lists available voices (OpenWebUI primary discovery)
  GET /v1/audio/voices  - Lists available voices (OpenWebUI fallback)
  GET /v1/audio/models  - Lists available voices (OpenWebUI fallback)
  GET /speakers         - Lists speaker IDs (SillyTavern)
  OPTIONS /{path}       - Pre-flight CORS handler

Startup:
  CUDA graphs are warmed up on server start so the first real request
  does not pay the 7-8s graph-compilation penalty.
"""

import asyncio
import logging
import sys
import os
import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# Point Python to the app directory inside the container
sys.path.append("/app/examples")
import openai_server

logger = logging.getLogger(__name__)


def _do_warmup():
    """Run one short generation to compile CUDA graphs before serving requests."""
    model = openai_server.tts_model
    voices = openai_server.voices
    default_voice = openai_server.default_voice

    if model is None or not voices or default_voice is None:
        logger.warning("Warmup skipped: model or voices not ready")
        return

    voice_cfg = voices.get(default_voice, {})
    ref_audio = voice_cfg.get("ref_audio")
    if not ref_audio:
        logger.warning("Warmup skipped: no ref_audio on default voice")
        return

    logger.info("Warming up CUDA graphs (first request will be fast)...")
    try:
        for _ in model.generate_voice_clone_streaming(
            text="Warmup.",
            language=voice_cfg.get("language", "Auto"),
            ref_audio=ref_audio,
            ref_text=voice_cfg.get("ref_text", ""),
            chunk_size=12,
            non_streaming_mode=True,
        ):
            pass
        logger.info("CUDA warmup complete — server ready.")
    except Exception as exc:
        logger.warning("Warmup failed (non-fatal): %s", exc)


def _precompute_all_embeddings():
    """Background task to precompute all missing embeddings to avoid lazy-load delay."""
    model = openai_server.tts_model
    voices = openai_server.voices
    if not model or not voices:
        return
        
    for voice_name, voice_cfg in list(voices.items()):
        # skip if already precomputed
        spk_emb_path = voice_cfg.get("speaker_embeddings") or voice_cfg.get("speaker embeddings")
        if spk_emb_path and os.path.isfile(spk_emb_path):
            continue
            
        logger.info("Background precomputing embedding for %r...", voice_name)
        # We must lock the model to prevent concurrent generation with incoming requests
        with openai_server._model_lock:
            try:
                # _load_voice_clone_prompt updates voice_cfg in place and saves the .pt
                openai_server._load_voice_clone_prompt(voice_cfg, voice_name, model)
            except Exception as e:
                logger.error("Failed to background precompute for %r: %s", voice_name, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_warmup)
    loop.run_in_executor(None, _precompute_all_embeddings)
    yield


# Attach lifespan to the existing FastAPI app
openai_server.app.router.lifespan_context = lifespan

# Load generated voices
try:
    with open('/config/voices.json', 'r') as f:
        voices_data = json.load(f)
except FileNotFoundError:
    voices_data = {}

# Build reusable response payloads
_voice_list = [{'id': v, 'object': 'model', 'created': 1686935002, 'owned_by': 'qwen'} for v in voices_data.keys()]
_models_response = {'object': 'list', 'data': _voice_list}

# OpenWebUI model discovery (primary)
@openai_server.app.get('/v1/models')
async def list_models():
    return _models_response

# OpenWebUI voice discovery fallbacks
@openai_server.app.get('/v1/audio/voices')
async def list_audio_voices():
    return _models_response

@openai_server.app.get('/v1/audio/models')
async def list_audio_models():
    return _models_response

# SillyTavern speaker endpoint
@openai_server.app.get('/speakers')
async def get_speakers():
    return list(voices_data.keys())

# Pre-flight OPTIONS handler to prevent 404s
@openai_server.app.options('/{path:path}')
async def options_handler(path: str):
    return JSONResponse(content={'status': 'ok'})

_voices_lock = threading.Lock()


@openai_server.app.post('/voice-seed')
async def set_voice_seed(request: Request):
    """Set (or clear) the seed for a voice in voices.json.

    Body: {"voice": "EN_F_NatashaNeural", "seed": 7}
    To remove a seed: {"voice": "EN_F_NatashaNeural", "seed": null}
    """
    data = await request.json()
    voice_name = data.get("voice")
    seed = data.get("seed")

    if not voice_name:
        raise HTTPException(status_code=400, detail="'voice' field is required")

    voices_path = '/config/voices.json'
    with _voices_lock:
        try:
            with open(voices_path, 'r', encoding='utf-8') as f:
                voices = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="voices.json not found")

        if voice_name not in voices:
            raise HTTPException(status_code=404, detail=f"Voice {voice_name!r} not found")

        if seed is None:
            voices[voice_name].pop("seed", None)
        else:
            try:
                voices[voice_name]["seed"] = int(seed)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="'seed' must be an integer or null")

        with open(voices_path, 'w', encoding='utf-8') as f:
            json.dump(voices, f, indent=2, ensure_ascii=False)

    return JSONResponse({"ok": True, "voice": voice_name, "seed": seed})


_SEED_SAMPLES_DIR = '/config/seed_samples'


@openai_server.app.get('/seed-samples/{voice_name}')
async def list_seed_samples(voice_name: str):
    """Return a sorted list of seed numbers for which a pre-generated WAV exists."""
    import re as _re
    voice_dir = os.path.join(_SEED_SAMPLES_DIR, voice_name)
    if not os.path.isdir(voice_dir):
        return JSONResponse({"seeds": []})
    seeds = []
    for fname in os.listdir(voice_dir):
        m = _re.match(r'^seed_(\d+)\.wav$', fname)
        if m:
            seeds.append(int(m.group(1)))
    seeds.sort()
    return JSONResponse({"seeds": seeds})


@openai_server.app.get('/seed-sample/{voice_name}/{seed}')
async def get_seed_sample(voice_name: str, seed: int):
    """Serve a pre-generated seed WAV file."""
    from fastapi.responses import FileResponse
    path = os.path.join(_SEED_SAMPLES_DIR, voice_name, f'seed_{seed:05d}.wav')
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No sample for seed {seed}")
    return FileResponse(path, media_type='audio/wav')


if __name__ == '__main__':
    openai_server.main()
