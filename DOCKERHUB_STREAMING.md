# Qwen3-TTS Streaming — DGX Spark (GB10)

Low-latency, OpenAI-compatible streaming TTS server for the **NVIDIA DGX Spark GB10**
(ARM64 / SM 121 / CUDA 13), powered by [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
with CUDA graph acceleration.

Streams WAV audio chunks to the client while generation is still running —
first audio arrives in under a second for typical sentences.

Part of a four-backend TTS stack documented on the NVIDIA Developer Forum:
[Three times (VoiceClone | VoiceDesign | CustomVoice) — Faster-Qwen3-TTS for NVIDIA DGX Spark (GB10)](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530)

---

## Quick start

```bash
docker run -d \
  --runtime nvidia \
  --name qwen3-tts-streaming \
  -p 8023:8000 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /path/to/Qwen3-TTS-12Hz-1.7B-Base:/models/Qwen3-TTS:ro \
  -v /path/to/faster-qwen3-tts/config:/config:rw \
  -v /path/to/active_voices:/voices:ro \
  martinb78/faster-qwen3-tts-dgx-spark:streaming \
  /bin/bash -c "
    python3 /config/generate_voices.py &&
    python3 /config/run_server.py
      --model /models/Qwen3-TTS
      --voices /config/voices.json
      --port 8000
      --max-seq-len 4096
  "
```

Check it's running:

```bash
curl http://localhost:8023/health
```

---

## Voice configuration

Create a `voices.json` in your config directory. Each entry maps a voice ID to a
reference audio file and transcript:

```json
{
  "william": {
    "ref_audio": "/voices/william.wav",
    "ref_text": "The quick brown fox jumps over the lazy dog.",
    "language": "English",
    "temperature": 0.75,
    "top_k": 40,
    "top_p": 0.85
  },
  "natasha": {
    "ref_audio": "/voices/natasha.wav",
    "ref_text": "She sells seashells by the seashore.",
    "language": "English"
  }
}
```

`temperature`, `top_k`, and `top_p` are optional — defaults are `0.8 / 50 / 0.9`.

---

## API

OpenAI-compatible `/v1/audio/speech`:

```bash
curl http://localhost:8023/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "Hello world!", "voice": "william", "response_format": "wav"}' \
  --output speech.wav
```

| Endpoint | Method | Description |
|---|---|---|
| `/v1/audio/speech` | POST | Generate speech (WAV / PCM / MP3) |
| `/v1/models` | GET | List available voice IDs |
| `/v1/audio/voices` | GET | Voice list (OpenWebUI fallback) |
| `/speakers` | GET | Voice list (SillyTavern) |
| `/health` | GET | Liveness check |

Works with **OpenWebUI**, **SillyTavern**, **llama-swap**, and any OpenAI-compatible client.

---

## Requirements

- NVIDIA DGX Spark GB10 or another ARM64 system with CUDA 13
- CUDA driver 580+
- Docker + NVIDIA Container Toolkit
- [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) weights downloaded locally

---

## Related images

| Image | Description |
|---|---|
| `martinb78/faster-qwen3-tts-dgx-spark:latest` / `:v5` | VoiceClone, VoiceDesign, and CustomVoice backends |
| `martinb78/faster-qwen3-tts-dgx-spark:streaming` | This tag — streaming VoiceClone |

Full four-backend `docker-compose` setup in `docker/docker-compose.yml` and detailed documentation on GitHub and the
[NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/three-times-voiceclone-voicedesign-customvoice-faster-qwen3-tts-for-nvidia-dgx-spark-gb10/370530).
