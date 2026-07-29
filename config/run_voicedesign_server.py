"""
OpenAI-compatible TTS server for Qwen3-TTS-12Hz-1.7B-VoiceDesign.

Voices are defined in voicedesign_voices.json as:
  { "voice_id": { "instruct": "...", "language": "..." } }

No ref_audio needed — the instruct text fully describes the voice.
"""
import json
import logging
import os
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
voices_file_path: str = None
last_voices_mtime: float = 0.0
SAMPLE_RATE = 24000
DEFAULT_MAX_NEW_TOKENS = 2048
_model_lock = threading.Lock()
_load_model_kwargs = None
aligner_model = None

def _get_aligner():
    global aligner_model
    if aligner_model is None:
        try:
            from qwen_asr import Qwen3ForcedAligner
            import torch
        except ImportError:
            raise HTTPException(status_code=500, detail="qwen-asr is not installed. Run: pip install qwen-asr")
        logger.info("Loading Qwen3-ForcedAligner-0.6B...")
        aligner_model = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B", 
            dtype=torch.bfloat16, 
            device_map="cuda"
        )
        logger.info("Aligner loaded.")
    return aligner_model

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
    response_format: str = "wav"  # wav | pcm | mp3 | zip
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


def _reload_voices_if_changed():
    """Pick up voices added to the registry since startup.

    The registry used to be read once in main(), so any voice designed while
    the server was up stayed invisible until a manual restart — and an unknown
    voice silently became a bundled preset (see resolve_voice). The voice-clone
    server already hot-reloads its registry; this brings VoiceDesign in line.
    """
    global voices, last_voices_mtime
    if not voices_file_path or not os.path.exists(voices_file_path):
        return
    try:
        mtime = os.path.getmtime(voices_file_path)
        if mtime > last_voices_mtime:
            with open(voices_file_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and loaded:
                voices = loaded
                last_voices_mtime = mtime
                _build_voice_list()
                logger.info("Hot-reloaded %d voices from %s", len(voices), voices_file_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to hot-reload voices: %s", exc)


def resolve_voice(name: str, has_request_instruct: bool = False) -> dict:
    _reload_voices_if_changed()
    cfg = voices.get(name)
    if cfg:
        return cfg
    # Falling back to another voice is meaningless here: for VoiceDesign the
    # instruct IS the voice, so substituting the first registered preset does
    # not degrade the result, it silently returns a completely different
    # character — confirmed live as the cause of a German male character being
    # read by 'vd_british_male', including apparent gender flips between takes.
    if has_request_instruct:
        # The caller described the voice inline, so nothing is missing.
        logger.info("Voice %r not registered; using the instruct supplied with the request", name)
        return {}
    raise HTTPException(
        status_code=404,
        detail=(
            f"Voice {name!r} is not registered and the request carried no 'instruct' "
            f"to describe it. Known voices: {sorted(voices)}"
        ),
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _request_generation_params(req: SpeechRequest, voice_cfg: dict) -> dict:
    # A registered voice's instruct is its IDENTITY; an instruct sent with the
    # request is a per-line direction (emotion, delivery). Replacing the former
    # with the latter — the previous behaviour — meant every line that carried
    # any emotion discarded the character's voice entirely and re-rolled a new
    # one from a few words of direction, which is why a character's voice
    # drifted between lines. Combine them, identity first.
    base_instruct = str(voice_cfg.get("instruct", "") or "").strip()
    line_instruct = str(req.instruct or "").strip()
    if base_instruct and line_instruct and line_instruct != base_instruct:
        instruct = f"{base_instruct} {line_instruct}"
    else:
        instruct = line_instruct or base_instruct

    language = req.language or voice_cfg.get("language", "English")
    params = {
        "text": req.input,
        "instruct": instruct,
        "language": language,
        "max_new_tokens": req.max_new_tokens or int(voice_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
    }
    # Per-voice sampling overrides — the only way to make a designed voice
    # reproducible, since generate_voice_design() accepts no seed.
    for key in ("temperature", "top_p", "top_k"):
        if voice_cfg.get(key) is not None:
            params[key] = float(voice_cfg[key]) if key != "top_k" else int(voice_cfg[key])
    return params


async def _stream_chunks(params: dict, speed: float):
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        process = None
        if speed != 1.0:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
                "-filter:a", f"atempo={speed}",
                "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            
            def ffmpeg_reader():
                try:
                    while True:
                        out = process.stdout.read(4096)
                        if not out:
                            break
                        q.put(out)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(_DONE)
                    
            import threading
            threading.Thread(target=ffmpeg_reader, daemon=True).start()

        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_design_streaming(**params):
                    raw = _to_pcm16(chunk)
                    if process:
                        process.stdin.write(raw)
                        process.stdin.flush()
                    else:
                        q.put(raw)
        except Exception as exc:
            q.put(exc)
        finally:
            if process:
                try:
                    process.stdin.close()
                except Exception:
                    pass
            else:
                q.put(_DONE)

    import threading
    threading.Thread(target=producer, daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield item


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
    req.input = req.input.strip()
    if not req.input:
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg = resolve_voice(req.voice, has_request_instruct=bool((req.instruct or "").strip()))
    params = _request_generation_params(req, voice_cfg)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {"wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg", "zip": "application/zip"}
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt!r}")

    if fmt in ("mp3", "zip"):
        loop = asyncio.get_event_loop()
        def _gen():
            with _model_lock:
                return tts_model.generate_voice_design(**params)
        audio_arrays, sr = await loop.run_in_executor(None, _gen)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
        
        if req.speed != 1.0:
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                "-filter:a", f"atempo={req.speed}",
                "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            process.stdin.write(audio.tobytes())
            process.stdin.close()
            out = process.stdout.read()
            audio = np.frombuffer(out, dtype=np.float32)

        if fmt == "zip":
            def _align():
                aligner = _get_aligner()
                res = aligner.align(audio=(audio, sr), text=req.input, language=voice_cfg.get("language", "Auto"))
                import dataclasses
                return [dataclasses.asdict(x) for x in res]
            
            align_data = await loop.run_in_executor(None, _align)
            
            import zipfile
            import io
            mp3_bytes = _to_mp3_bytes(audio, sr)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("audio.mp3", mp3_bytes)
                zf.writestr("timer.json", json.dumps(align_data, ensure_ascii=False))
                
            return Response(content=zip_buf.getvalue(), media_type=_CONTENT_TYPES[fmt])

        return Response(content=_to_mp3_bytes(audio, sr), media_type="audio/mpeg")

    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)
        async for raw in _stream_chunks(params, req.speed):
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
    _reload_voices_if_changed()
    return _models_response

@app.get("/v1/audio/voices")
async def list_audio_voices():
    _reload_voices_if_changed()
    return _models_response

@app.get("/v1/audio/models")
async def list_audio_models():
    _reload_voices_if_changed()
    return _models_response

@app.get("/speakers")
async def get_speakers():
    _reload_voices_if_changed()
    return list(voices.keys())

@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(content={"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global voices, default_voice, SAMPLE_RATE, DEFAULT_MAX_NEW_TOKENS, _load_model_kwargs
    global voices_file_path, last_voices_mtime

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

    voices_file_path = args.voices
    # The registry is generated by the voice-clone container and is not tracked
    # in git, so on a fresh checkout it may not exist yet. Start from the
    # bundled presets instead of crashing — hot-reload picks up the real
    # registry as soon as it appears.
    startup_file = args.voices
    if not os.path.exists(startup_file):
        startup_file = os.path.join(
            os.path.dirname(args.voices) or ".", "voicedesign_voices.presets.json"
        )
        logger.warning(
            "%s not found, falling back to bundled presets %s",
            args.voices, startup_file,
        )
    with open(startup_file) as f:
        voices = json.load(f)
    try:
        last_voices_mtime = os.path.getmtime(args.voices)
    except OSError:
        last_voices_mtime = 0.0
    default_voice = next(iter(voices), None)
    _build_voice_list()
    logger.info("Loaded %d voices from %s", len(voices), startup_file)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
