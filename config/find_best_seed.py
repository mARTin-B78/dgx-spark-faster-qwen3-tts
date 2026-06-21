#!/usr/bin/env python3
"""
Find the best RNG seed for a voice by generating audio samples with different
seeds and saving them as numbered WAV files for comparison.

How it works:
  1. For each seed, temporarily writes that seed into voices.json
  2. Calls the running server (hot-reload picks it up automatically)
  3. Saves the audio as seed_0001.wav, seed_0042.wav, etc.
  4. Restores voices.json to its original state when done

Then just listen to the WAV files and pick the seed number you prefer.
Add it to your voice in voices.json: "seed": <number>

Usage:
    python find_best_seed.py --voice EN_F_NatashaNeural --seeds 1 2 3 42 100
    python find_best_seed.py --voice EN_F_NatashaNeural --range 1 20
    python find_best_seed.py --voice EN_F_NatashaNeural --range 1 50 --text "Hello!"

    # Different server port:
    python find_best_seed.py --voice DE_5_28 --range 1 10 --port 8020
"""
import argparse
import json
import os
import shutil
import sys
import time

import requests

VOICES_JSON = "/config/voices.json"
DEFAULT_TEXT = (
    "The morning light filtered softly through the curtains, casting a warm glow "
    "across the room. Outside, birds were already singing."
)


def load_voices() -> dict:
    with open(VOICES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_voices(voices: dict) -> None:
    with open(VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)


def generate(voice: str, text: str, host: str, port: int, timeout: int = 60) -> bytes:
    url = f"http://{host}:{port}/v1/audio/speech"
    resp = requests.post(
        url,
        json={"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"},
        timeout=timeout,
        stream=False,
    )
    resp.raise_for_status()
    return resp.content


def main():
    p = argparse.ArgumentParser(description="Compare seeds for a voice")
    p.add_argument("--voice", required=True, help="Voice name (must exist in voices.json)")
    p.add_argument("--seeds", type=int, nargs="+", help="Explicit list of seeds to try")
    p.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                   help="Try seeds START through END (inclusive)")
    p.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--out-dir", default="./seed_samples", help="Directory to save WAV files")
    args = p.parse_args()

    if not args.seeds and not args.range:
        print("ERROR: provide --seeds or --range", file=sys.stderr)
        sys.exit(1)

    seeds = list(args.seeds or [])
    if args.range:
        seeds += list(range(args.range[0], args.range[1] + 1))
    seeds = sorted(set(seeds))

    os.makedirs(args.out_dir, exist_ok=True)

    # Back up voices.json
    backup = VOICES_JSON + ".seed_backup"
    shutil.copy2(VOICES_JSON, backup)
    print(f"Backed up voices.json → {backup}")

    voices = load_voices()
    if args.voice not in voices:
        print(f"ERROR: voice {args.voice!r} not found in voices.json", file=sys.stderr)
        print(f"Available: {', '.join(voices.keys())}", file=sys.stderr)
        sys.exit(1)

    original_seed = voices[args.voice].get("seed", "<none>")
    print(f"\nVoice:    {args.voice}")
    print(f"Text:     {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
    print(f"Seeds:    {seeds}")
    print(f"Output:   {os.path.abspath(args.out_dir)}/")
    print(f"Original seed: {original_seed}\n")

    results = []

    try:
        for seed in seeds:
            voices[args.voice]["seed"] = seed
            save_voices(voices)
            # Brief pause so the server's hot-reload detects the mtime change
            time.sleep(0.3)

            out_path = os.path.join(args.out_dir, f"seed_{seed:05d}.wav")
            print(f"  seed {seed:5d} → ", end="", flush=True)
            try:
                t0 = time.time()
                wav = generate(args.voice, args.text, args.host, args.port)
                elapsed = time.time() - t0
                with open(out_path, "wb") as f:
                    f.write(wav)
                print(f"{out_path}  ({elapsed:.1f}s)")
                results.append((seed, out_path, None))
            except Exception as e:
                print(f"FAILED: {e}")
                results.append((seed, None, str(e)))

    finally:
        # Always restore the original voices.json
        shutil.copy2(backup, VOICES_JSON)
        os.remove(backup)
        print(f"\nRestored voices.json (seed reset to {original_seed!r})")

    print(f"\n{'─'*60}")
    print("Done! Listen to the files and find the seed you prefer.")
    print(f"Then add it to voices.json under {args.voice!r}:")
    print(f'    "seed": <your_chosen_number>')
    print(f"{'─'*60}")
    ok = [(s, p) for s, p, e in results if p]
    if ok:
        print(f"\nGenerated {len(ok)} samples:")
        for seed, path in ok:
            print(f"  seed {seed:5d}  →  {path}")


if __name__ == "__main__":
    main()
