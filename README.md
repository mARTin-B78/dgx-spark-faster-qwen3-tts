# Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10)

Run [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) on the **NVIDIA DGX Spark GB10** (ARM64 / SM 121 / CUDA 13) as a persistent, OpenAI-compatible TTS API.

![Faster-Qwen3-TTS on NVIDIA DGX Spark](https://global.discourse-cdn.com/nvidia/original/4X/5/8/0/58098a94620f87839a47638804ecff6c2c554211.png)

This repo packages the DGX Spark fixes plus API servers for three Qwen3-TTS modes:

| Mode | Port | Model | Voice source |
|---|---:|---|---|
| VoiceClone | `8020` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Reference audio plus transcript |
| VoiceDesign | `8021` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Plain-English voice instructions |
| CustomVoice | `8022` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Built-in speaker IDs |

All modes expose the OpenAI `/v1/audio/speech` contract and work with **OpenWebUI**, **SillyTavern**, **llama-swap**, `curl`, or any OpenAI-compatible client.

## What this solves

The DGX Spark GB10 has a unique ARM64 Grace CPU plus Blackwell GPU stack (SM 121 / CUDA 13). Standard ML containers often need small but important changes:

- **torchaudio ARM64 wheels** - resolved by using PyTorch's `cu130` wheel index.
- **Flash Attention on SM 121** - avoided; faster-qwen3-tts uses CUDA graphs instead.
- **CUDA graph capture** - configured for low-latency Qwen3-TTS inference.
- **OpenAI compatibility** - `/v1/audio/speech`, `/v1/models`, `/v1/audio/voices`, `/v1/audio/models`, and `/speakers` are available for common clients.

## Quick start: VoiceClone only

Use the root `docker-compose.yml` when you only need voice cloning on port `8020`.

```bash
docker pull martinb78/faster-qwen3-tts-dgx-spark:latest

mkdir -p models
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS

cp .env.example .env
# Edit .env and set MODEL_PATH to your local Qwen3-TTS-12Hz-1.7B-Base directory.

# Add reference audio and transcripts to config/speakers/ first.
docker compose up -d
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

## Full stack: VoiceClone, VoiceDesign, CustomVoice

Use `config/docker-compose.yml` when you want all Qwen3-TTS modes side by side. The file also includes an optional low-latency streaming VoiceClone service on port `8023`; remove or comment that service if you only want the three main endpoints.

1. Download the models you want to run:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

2. Edit `config/docker-compose.yml` and adjust the volume paths for your machine:

```yaml
volumes:
  - /path/to/Qwen3-TTS-12Hz-1.7B-Base:/models/Qwen3-TTS:ro
  - /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign:/models/Qwen3-TTS-VoiceDesign:ro
  - /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice:/models/Qwen3-TTS-CustomVoice:ro
  - /path/to/this/repo/config:/config:rw
```

3. Make sure the external Docker network exists, then start the stack:

```bash
docker network create dgx_net 2>/dev/null || true
cd config
docker compose up -d
```

4. Check the services:

```bash
curl http://localhost:8020/health   # VoiceClone
curl http://localhost:8021/health   # VoiceDesign
curl http://localhost:8022/health   # CustomVoice
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

Per-request fields win over the JSON voice config entry, so one configured voice can still be adjusted by callers for language, tone, or generation length.

## Client configuration

### OpenWebUI

In OpenWebUI Settings > Audio > Text-to-Speech:

| Setting | Value |
|---|---|
| Engine | OpenAI |
| URL | `http://your-host:8020/v1`, `http://your-host:8021/v1`, or `http://your-host:8022/v1` |
| API Key | `sk-dummy-key` |
| TTS Model | `tts-1` |
| TTS Voice | Select from dropdown |

### llama-swap or other OpenAI-compatible clients

Point the client's OpenAI-compatible TTS base URL at the service you want:

```text
http://your-host:8020/v1   # VoiceClone
http://your-host:8021/v1   # VoiceDesign
http://your-host:8022/v1   # CustomVoice
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
- The forum playbook shows the three API containers running together on DGX Spark with low visible memory pressure, but exact usage depends on model size, sequence length, and warmup state.
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
- Docker plus NVIDIA Container Toolkit.
- Local Qwen3-TTS model weights from Hugging Face.

## Credits

- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by Andres Marafioti.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team.
- DGX Spark compatibility, Docker, and OpenAI-compatible API packaging by [mARTin-B78](https://github.com/mARTin-B78).
- NVIDIA Developer Forum playbook: [Three times (VoiceClone | VoiceDesign | CustomVoice) - Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10)](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530).

## License

MIT (same as upstream faster-qwen3-tts).
