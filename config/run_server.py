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
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _do_warmup)
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

if __name__ == '__main__':
    openai_server.main()
