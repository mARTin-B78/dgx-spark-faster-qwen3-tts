#!/usr/bin/env python3
"""
Benchmark the faster-qwen3-tts API server.

Measures:
  - TTFA  : time-to-first-audio (ms) — latency before playback can start
  - Total : wall-clock time until full audio received (s)
  - RTF   : real-time factor = generation_time / audio_duration  (lower = faster)
  - Speed : audio_duration / generation_time  (higher = faster, e.g. 3× real-time)

Usage:
    python benchmark_api.py [--host localhost] [--port 8020] [--voice alloy]
                            [--runs 3] [--format wav]
"""
import argparse
import struct
import sys
import time

import requests

# ── Test sentences of increasing length ─────────────────────────────────────
SENTENCES = [
    ("short",  "Hello, how are you today?"),
    ("medium", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("long",   "Artificial intelligence is transforming the way we interact with "
               "technology. From voice assistants to autonomous vehicles, machine "
               "learning models are becoming an integral part of everyday life."),
]

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # 16-bit PCM


def parse_wav_header(data: bytes) -> int:
    """Return the data-chunk offset so we can skip the header bytes."""
    # WAV: RIFF(4) + size(4) + WAVE(4) + fmt (8+16) + data(4) + size(4) = 44 bytes
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0
    # Walk chunks after WAVE marker
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos+4]
        chunk_sz = struct.unpack_from("<I", data, pos+4)[0]
        if chunk_id == b"data":
            return pos + 8
        pos += 8 + chunk_sz
    return 44  # fallback


def benchmark_request(url: str, payload: dict, label: str, run: int) -> dict:
    t0 = time.perf_counter()
    first_byte_t = None
    raw = bytearray()

    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=512):
            if chunk:
                if first_byte_t is None:
                    first_byte_t = time.perf_counter()
                raw.extend(chunk)

    t1 = time.perf_counter()

    ttfa_ms    = (first_byte_t - t0) * 1000 if first_byte_t else float("nan")
    total_s    = t1 - t0

    # Strip WAV header to count actual PCM bytes
    if payload.get("response_format", "wav") == "wav":
        data_offset = parse_wav_header(bytes(raw))
        pcm_bytes = len(raw) - data_offset
    else:
        pcm_bytes = len(raw)

    audio_s  = pcm_bytes / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    rtf      = total_s / audio_s if audio_s > 0 else float("nan")
    speedup  = audio_s / total_s if total_s > 0 else float("nan")

    return {
        "label":    label,
        "run":      run,
        "ttfa_ms":  ttfa_ms,
        "total_s":  total_s,
        "audio_s":  audio_s,
        "rtf":      rtf,
        "speedup":  speedup,
    }


def main():
    p = argparse.ArgumentParser(description="Benchmark faster-qwen3-tts API")
    p.add_argument("--host",   default="localhost")
    p.add_argument("--port",   type=int, default=8020)
    p.add_argument("--voice",  default=None,
                   help="Voice name (default: first available voice)")
    p.add_argument("--runs",   type=int, default=3,
                   help="Repetitions per sentence (default: 3)")
    p.add_argument("--format", default="wav", choices=["wav", "pcm"],
                   help="Response format (default: wav)")
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    url  = f"{base}/v1/audio/speech"

    # Auto-detect first voice if not specified
    voice = args.voice
    if not voice:
        try:
            models = requests.get(f"{base}/v1/models", timeout=5).json()
            voice = models["data"][0]["id"]
            print(f"Auto-selected voice: {voice!r}")
        except Exception:
            voice = "alloy"
            print(f"Could not auto-detect voice, using {voice!r}")

    print(f"\n{'─'*65}")
    print(f"  faster-qwen3-tts API Benchmark")
    print(f"  URL: {url}  voice={voice}  runs={args.runs}  fmt={args.format}")
    print(f"{'─'*65}\n")

    header = f"{'Label':<10}  {'Run':>3}  {'TTFA*':>9}  {'Total':>7}  {'Audio':>7}  {'RTF':>6}  {'Speed':>8}"
    print(header)
    print("─" * len(header))
    print("  * TTFA = time to first PCM byte (WAV header excluded)")
    print()

    all_results = []

    for label, text in SENTENCES:
        payload = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": args.format,
        }
        for run in range(1, args.runs + 1):
            try:
                r = benchmark_request(url, payload, label, run)
                all_results.append(r)
                print(
                    f"{r['label']:<10}  {r['run']:>3}  "
                    f"{r['ttfa_ms']:>7.0f}ms  "
                    f"{r['total_s']:>6.2f}s  "
                    f"{r['audio_s']:>6.2f}s  "
                    f"{r['rtf']:>5.2f}x  "
                    f"{r['speedup']:>6.1f}x rt"
                )
                sys.stdout.flush()
            except Exception as e:
                print(f"{label:<10}  run {run}  ERROR: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    if all_results:
        print(f"\n{'─'*65}")
        print("  Summary (averages across all runs)")
        print(f"{'─'*65}")
        for label, _ in SENTENCES:
            rows = [r for r in all_results if r["label"] == label]
            if not rows:
                continue
            avg_ttfa  = sum(r["ttfa_ms"] for r in rows) / len(rows)
            avg_total = sum(r["total_s"] for r in rows) / len(rows)
            avg_rtf   = sum(r["rtf"]     for r in rows) / len(rows)
            avg_speed = sum(r["speedup"] for r in rows) / len(rows)
            print(
                f"  {label:<10}  TTFA={avg_ttfa:.0f}ms  "
                f"total={avg_total:.2f}s  RTF={avg_rtf:.2f}x  "
                f"speed={avg_speed:.1f}× real-time"
            )
        print()


if __name__ == "__main__":
    main()
