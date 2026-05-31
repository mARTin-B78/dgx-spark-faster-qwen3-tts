# Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10)

Run [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) on the **NVIDIA DGX Spark GB10** (ARM64 / SM 121 / CUDA 13) as a persistent, OpenAI-compatible TTS API.

![Faster-Qwen3-TTS on NVIDIA DGX Spark](https://global.discourse-cdn.com/nvidia/original/4X/5/8/0/58098a94620f87839a47638804ecff6c2c554211.png)

This repo packages the DGX Spark fixes plus four OpenAI-compatible TTS backends:

| Backend | Port | Image | Voice source |
|---|---:|---|---|
| VoiceClone | `8020` | `martinb78/faster-qwen3-tts-dgx-spark:latest` | Reference audio plus transcript |
| VoiceDesign | `8021` | `martinb78/faster-qwen3-tts-dgx-spark:latest` | Text prompt describes the voice; no reference needed |
| CustomVoice | `8022` | `martinb78/faster-qwen3-tts-dgx-spark:latest` | Separate CustomVoice model variant |
| Streaming | `8023` | `martinb78/faster-qwen3-tts-dgx-spark:latest-streaming` | Same voices as `8020`, but streams WAV chunks while generating |

All four backends expose the OpenAI `/v1/audio/speech` contract and work with **OpenWebUI**, **SillyTavern**, **llama-swap**, `curl`, or any OpenAI-compatible client.

One Docker image covers all four backends, with two semantic tag aliases:

| Tag | Use for |
|---|---|
| `:latest` / `:v6` | VoiceClone, VoiceDesign, CustomVoice |
| `:latest-streaming` / `:v6-streaming` | Streaming VoiceClone |

Both tags point to the same image — the `-streaming` suffix is a semantic convention so compose files and version pins are unambiguous.

## What this solves

The DGX Spark GB10 has a unique ARM64 Grace CPU plus Blackwell GPU stack (SM 121 / CUDA 13). Standard ML containers often need small but important changes:

- **torchaudio ARM64 wheels** - resolved by using PyTorch's `cu130` wheel index.
- **Flash Attention on SM 121** - avoided; faster-qwen3-tts uses CUDA graphs instead.
- **CUDA graph capture** - configured for low-latency Qwen3-TTS inference.
- **OpenAI compatibility** - `/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`, `/v1/audio/models`, and `/speakers` are available for common clients.

## Quick start: VoiceClone only

Use `docker/docker-compose.simple.yml` when you only need voice cloning on port `8020`.

```bash
docker pull martinb78/faster-qwen3-tts-dgx-spark:latest

mkdir -p models
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS

# Add reference audio and transcripts to config/speakers/ first.
cd docker
MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-Base docker compose -f docker-compose.simple.yml up -d
```

Build the image locally instead of pulling Docker Hub:

```bash
docker build -t faster-qwen3-tts-dgx-spark:latest .
```

If `docker compose up` reports that `dgx_net` is missing, create it once:

```bash
docker network create dgx_net
```

Check the server:

```bash
curl http://localhost:8020/health
```

## Full stack: VoiceClone, VoiceDesign, CustomVoice, Streaming

Use `docker/docker-compose.yml` when you want all four OpenAI-compatible backends side by side:

```text
8020  ->  VoiceClone   (/v1/audio/speech, reference audio)
8021  ->  VoiceDesign  (text prompt describes the voice, no reference needed)
8022  ->  CustomVoice  (separate CustomVoice model variant)
8023  ->  Streaming    (same as 8020 but streams WAV chunks while generating)
```

1. Download the models you want to run:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

2. Edit `docker/docker-compose.yml` and adjust the volume paths for your machine:

```yaml
volumes:
  - /path/to/Qwen3-TTS-12Hz-1.7B-Base:/models/Qwen3-TTS:ro
  - /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign:/models/Qwen3-TTS-VoiceDesign:ro
  - /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice:/models/Qwen3-TTS-CustomVoice:ro
  - /path/to/faster-qwen3-tts/config:/config:rw
```

3. Make sure the external Docker network exists, then start the stack:

```bash
docker network create dgx_net 2>/dev/null || true
cd docker
docker compose up -d
```

4. Check the services:

```bash
curl http://localhost:8020/health   # VoiceClone
curl http://localhost:8021/health   # VoiceDesign
curl http://localhost:8022/health   # CustomVoice
curl http://localhost:8023/health   # Streaming VoiceClone
```

## Adding VoiceClone voices

Place reference audio files in `config/speakers/` using this naming convention:

```text
EN_M_Speaker_Name.wav    # English, male
EN_F_Speaker_Name.wav    # English, female
DE_M_Speaker_Name.wav    # German, male
```

Reference audio should be **5-15 seconds** long. Longer files can slow inference and reduce cloning quality.

For each audio file, create a matching transcript:

```text
EN_M_Speaker_Name.reference.txt
```

Or use the auto-transcription script with a running Whisper-compatible ASR service:

```bash
python config/auto_transcribe.py --api-url http://localhost:8010/v1/audio/transcriptions
```

`config/generate_voices.py` runs on container startup and creates `config/voices.json` from your speaker files.

## VoiceDesign voices

VoiceDesign does not need reference audio. Define reusable voice personalities in `config/voicedesign_voices.json`:

```json
{
  "narrator": {
    "instruct": "Warm, confident narrator with a slight British accent",
    "language": "English"
  },
  "assistant_de": {
    "instruct": "Freundliche, klare Sprecherin, Hochdeutsch, professionell",
    "language": "German"
  }
}
```

Then call the VoiceDesign service on port `8021`.

## CustomVoice speakers

CustomVoice uses the model's built-in speaker names. Define the speaker IDs you want to expose in `config/customvoice_voices.json`:

```json
{
  "Ryan": {
    "speaker": "Ryan",
    "language": "English",
    "instruct": ""
  },
  "Ono_Anna": {
    "speaker": "Ono_Anna",
    "language": "Japanese",
    "instruct": ""
  },
  "Sohee": {
    "speaker": "Sohee",
    "language": "Korean",
    "instruct": ""
  }
}
```

Then call the CustomVoice service on port `8022`.

## Streaming backend

The streaming service on port `8023` uses the same generated `config/voices.json` and active VoiceClone reference voices as port `8020`, but returns WAV chunks while generation is still running. Use it when time-to-first-audio matters more than waiting for the complete WAV response.

## API

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/audio/speech` | POST | Generate speech in OpenAI-compatible format |
| `/v1/models` | GET | List available voice IDs |
| `/v1/audio/voices` | GET | OpenWebUI voice-list fallback |
| `/v1/audio/models` | GET | OpenWebUI model-list fallback |
| `/speakers` | GET | Speaker IDs for SillyTavern and simple clients |

### Speech request fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `tts-1` | Kept for OpenAI compatibility |
| `input` | string | required | Text to synthesize |
| `voice` | string | first configured voice | Voice ID from the selected service |
| `response_format` | string | `wav` | `wav`, `pcm`, or `mp3` |
| `language` | string | voice config | Per-request override for VoiceDesign/CustomVoice |
| `instruct` | string | voice config | Per-request style override for VoiceDesign/CustomVoice |
| `max_new_tokens` | int | server default | Per-request generation length override |

WAV and PCM are streamed as audio is generated. MP3 is encoded after generation and returned as a complete response.

### Examples

VoiceClone on port `8020`:

```bash
curl http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world!","voice":"EN_M_Speaker_Name","response_format":"wav"}' \
  --output speech.wav
```

VoiceDesign on port `8021`:

```bash
curl http://localhost:8021/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Welcome to the show.","voice":"narrator"}' \
  --output speech.wav
```

Per-request VoiceDesign override:

```bash
curl http://localhost:8021/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Herzlich willkommen.",
    "voice": "narrator",
    "language": "German",
    "instruct": "Speak slowly and warmly.",
    "max_new_tokens": 1024
  }' \
  --output speech_de.wav
```

CustomVoice on port `8022`:

```bash
curl http://localhost:8022/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This uses a built-in Qwen3-TTS speaker.","voice":"Ryan"}' \
  --output customvoice.wav
```

Streaming VoiceClone on port `8023`:

```bash
curl http://localhost:8023/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"This starts playing as chunks arrive.","voice":"EN_M_Speaker_Name","response_format":"wav"}' \
  --output streaming.wav
```

Per-request fields win over the JSON voice config entry, so one configured voice can still be adjusted by callers for language, tone, or generation length.

## Client configuration

### OpenWebUI

In OpenWebUI Settings > Audio > Text-to-Speech:

| Setting | Value |
|---|---|
| Engine | OpenAI |
| URL | `http://your-host:8020/v1`, `http://your-host:8021/v1`, `http://your-host:8022/v1`, or `http://your-host:8023/v1` |
| API Key | `sk-dummy-key` |
| TTS Model | `tts-1` |
| TTS Voice | Select from dropdown |

### llama-swap or other OpenAI-compatible clients

Point the client's OpenAI-compatible TTS base URL at the service you want:

```text
http://your-host:8020/v1   # VoiceClone
http://your-host:8021/v1   # VoiceDesign
http://your-host:8022/v1   # CustomVoice
http://your-host:8023/v1   # Streaming VoiceClone
```

## Benchmarking

Use `config/benchmark_api.py` to verify latency and real-time performance:

```bash
python config/benchmark_api.py --host localhost --port 8021 --runs 5
```

The benchmark reports:

| Metric | Meaning |
|---|---|
| TTFA | Time to first audio byte; useful for interactive playback latency |
| RTF | Generation time divided by audio duration; lower is better |
| Speed | Audio duration divided by generation time; higher than `1.0x` is faster than real time |

The first request after container startup can be slower because CUDA graph capture runs once during warmup. Later requests should use the captured graph.

## Performance and memory notes

- The 1.7B Qwen3-TTS models use about 6 GB of GPU memory each in bfloat16.
- The forum playbook shows the four API containers running together on DGX Spark with low visible memory pressure, but exact usage depends on model size, sequence length, and warmup state.
- Use the 0.6B Qwen3-TTS variants if you want a lighter multi-service setup.
- `--max-seq-len 2048` handles most sentence-style TTS requests. Long-form narration may need `4096`, with more memory required.
- Pin services to different GPUs with `NVIDIA_VISIBLE_DEVICES=0`, `NVIDIA_VISIBLE_DEVICES=1`, and so on if your system has more than one GPU.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `503 Model not loaded` | Server still loading or warming up | Wait 30-60 seconds and check container logs |
| `404 Voice not found` | Voice ID is not in the JSON config | Check spelling or call `/speakers` |
| Very high TTFA | CUDA graph capture failed or fallback path is active | Check logs, reduce `--max-seq-len`, then restart |
| MP3 output error | MP3 dependencies are missing or ffmpeg is unavailable | Use `wav`/`pcm` or rebuild the image with MP3 support |
| OpenWebUI has no voices | Client cannot read the voice list | Confirm `/v1/models` and `/v1/audio/voices` are reachable from OpenWebUI |

## Hardware requirements

- NVIDIA DGX Spark GB10, or another ARM64 + NVIDIA GPU setup with CUDA 13 support.
- CUDA driver 580+ with CUDA 13.0 support.
- Docker plus NVIDIA Container Toolkit. Make sure you have configured the runtime: `sudo nvidia-ctk runtime configure --runtime=docker` and restarted the Docker daemon.
- Local Qwen3-TTS model weights from Hugging Face.

## Changelog

### v6.2 — 2026-05-30
**Fix: voice drift on streaming port 8023 + per-voice temperature persists across restarts**

- Streaming service (port 8023) was using the pre-v5 image (`martinb78/faster-qwen3-tts-dgx-spark:streaming`) which still had `non_streaming_mode=False` — the same voice drift bug as voiceclone had before v5. Fixed by switching streaming to `:latest` which carries the v5 patch. No image rebuild needed; compose-only change.
- `config/generate_voices.py` previously overwrote `voices.json` completely on every container start, discarding any manually added `temperature`, `top_k`, or `top_p` fields. Now merges from the existing `voices.json` so user-added sampling parameters survive restarts.

### v6 — 2026-05-30
**Restore `--max-seq-len`, consolidate Docker files, merge streaming repo**

- Restored `--max-seq-len` support in VoiceClone: upstream `openai_server.py` gained this argument after v5 was built; patch regenerated against current upstream to add it and wire it to `FasterQwen3TTS.from_pretrained()`
- Moved full 4-service compose from `config/docker-compose.yml` → `docker/docker-compose.yml`
- Moved single-service quickstart from root `docker-compose.yml` → `docker/docker-compose.simple.yml`
- Merged streaming image into `martinb78/faster-qwen3-tts-dgx-spark:streaming` tag; removed separate `qwen3-tts-streaming-dgx-spark` repository
- Removed `|| true` from Dockerfile `git apply` step so patch failures fail the build loudly

### v5 — 2026-05-30
**Fix: voice drifts and gender changes on long paragraphs (VoiceClone)**

The VoiceClone server was using `non_streaming_mode=False`, a mode designed for streaming LLM→TTS pipelines where the text arrives token by token. In this mode only **one text token** enters the model's KV cache during prefill; the rest are fed one-per-codec-step via a `trailing_text_hiddens` tensor. For a typical 54-word paragraph that tensor holds ~49 steps (~4 seconds of guidance) while the actual speech takes ~18 seconds — leaving **77 % of the audio generated with no text conditioning at all**. The model free-runs for that portion and drifts away from the reference voice, sometimes changing gender entirely.

The fix is to use `non_streaming_mode=True` (already the default for VoiceDesign and CustomVoice), which puts the full text in the prefill so the model can attend to it throughout generation. Temperature was also lowered from 0.9 to 0.8 and nucleus sampling (`top_p=0.9`) added to reduce accumulated stochasticity over long runs. All three parameters are now per-voice configurable in `voices.json`.

Changes:
- `patches/openai_server.patch` updated: VoiceClone streaming and MP3 paths now use `non_streaming_mode=True`
- `config/run_server.py`: warmup call aligned to `non_streaming_mode=True`
- Temperature default 0.9 → 0.8; `top_p=0.9` added; both overridable per voice via `voices.json`

### v4 — 2026-05-24
- Add async model loading and CUDA warmup for VoiceDesign and CustomVoice servers.
- Replace test beep with real William and Natasha voice samples.

### v3 — earlier
- Add streaming TTS backend (port 8023).
- Add CustomVoice server, benchmark tool, and VoiceDesign API improvements.
- Add multi-source voice pipeline with VoiceDesign support.

### v2 — earlier
- Reduce latency: CUDA warmup, chunk_size=4, max-seq-len 2048.
- Initial DGX Spark (GB10 / ARM64 / CUDA 13) packaging.

## Credits

- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by Andres Marafioti.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team.
- DGX Spark compatibility, Docker, and OpenAI-compatible API packaging by [mARTin-B78](https://github.com/mARTin-B78).
- NVIDIA Developer Forum playbook and source for the four-backend layout: [Three times (VoiceClone | VoiceDesign | CustomVoice) - Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10)](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530).

## License

MIT (same as upstream faster-qwen3-tts).
