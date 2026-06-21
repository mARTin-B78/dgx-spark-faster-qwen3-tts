#!/usr/bin/env python3
"""
Find the best RNG seed for one or all voices by generating audio samples with
different seeds and saving them as numbered WAV files for comparison.

How it works:
  1. For each voice × seed, temporarily sets that seed in voices.json
  2. Calls the running server (hot-reload picks it up automatically)
  3. Saves audio as <out-dir>/<voice-name>/seed_<N>.wav
  4. Restores voices.json to its original state when done

Then listen to the files and pick the seed you prefer.
Add it to voices.json: "seed": <number>

Usage:
    # All voices, seeds 1–15:
    python find_best_seed.py --all-voices --range 1 15 --port 8020

    # Single voice:
    python find_best_seed.py --voice EN_F_NatashaNeural --range 1 20
    python find_best_seed.py --voice DE_5_28 --seeds 1 7 42 100
"""
import argparse
import json
import os
import shutil
import sys
import time

import requests

VOICES_JSON = "/config/voices.json"

DE_TEXT = (
    "Die 3.500 neuen High-End Geräte für das Server-Update benötigen eine "
    "außergewöhnlich starke Kühlung und regelmäßige Maßnahmen, um die Performance "
    "bei großer Last zu gewährleisten."
)
EN_TEXT = (
    "The system administrator successfully configured the customized Docker stacks "
    "and benchmarked the inference engines at exactly 8:45 AM."
)
MIXED_TEXT = DE_TEXT + " - " + EN_TEXT


def load_voices() -> dict:
    with open(VOICES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_voices(voices: dict) -> None:
    with open(VOICES_JSON, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)


def generate(voice: str, text: str, host: str, port: int, timeout: int = 90) -> bytes:
    url = f"http://{host}:{port}/v1/audio/speech"
    resp = requests.post(
        url,
        json={"model": "tts-1", "input": text, "voice": voice, "response_format": "wav"},
        timeout=timeout,
        stream=False,
    )
    resp.raise_for_status()
    return resp.content


def pick_text(voice_name: str, override: str | None) -> str:
    if override:
        return override
    name_lower = voice_name.lower()
    if name_lower.startswith("de_"):
        return MIXED_TEXT
    if name_lower.startswith("en_") or name_lower.startswith("gb_"):
        return EN_TEXT
    return MIXED_TEXT


def run_voice(voice_name: str, voices: dict, seeds: list[int],
              text: str, host: str, port: int, out_dir: str) -> list[tuple]:
    voice_dir = os.path.join(out_dir, voice_name)
    os.makedirs(voice_dir, exist_ok=True)

    results = []
    for seed in seeds:
        voices[voice_name]["seed"] = seed
        save_voices(voices)
        time.sleep(0.3)  # let hot-reload detect mtime change

        out_path = os.path.join(voice_dir, f"seed_{seed:05d}.wav")
        print(f"    seed {seed:5d} → ", end="", flush=True)
        try:
            t0 = time.time()
            wav = generate(voice_name, text, host, port)
            elapsed = time.time() - t0
            with open(out_path, "wb") as f:
                f.write(wav)
            print(f"OK  ({elapsed:.1f}s)")
            results.append((seed, out_path, None))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((seed, None, str(e)))

    # Remove the temporary seed so the voice returns to its auto-seed
    voices[voice_name].pop("seed", None)
    save_voices(voices)

    return results


def main():
    p = argparse.ArgumentParser(description="Compare seeds for one or all voices")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--voice", help="Single voice name")
    target.add_argument("--all-voices", action="store_true", help="Run for every voice in voices.json")

    p.add_argument("--seeds", type=int, nargs="+", help="Explicit list of seeds to try")
    p.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                   help="Try seeds START through END (inclusive). Default: 1–15")
    p.add_argument("--text", default=None,
                   help="Override text (default: auto-picks DE/EN/mixed based on voice name)")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--out-dir", default="./seed_samples",
                   help="Root output directory (voice subdirs created inside)")
    args = p.parse_args()

    seeds = list(args.seeds or [])
    if args.range:
        seeds += list(range(args.range[0], args.range[1] + 1))
    if not seeds:
        seeds = list(range(1, 16))  # default 1–15
    seeds = sorted(set(seeds))

    voices = load_voices()

    if args.voice:
        voice_names = [args.voice]
        if args.voice not in voices:
            print(f"ERROR: voice {args.voice!r} not in voices.json", file=sys.stderr)
            sys.exit(1)
    else:
        voice_names = list(voices.keys())

    os.makedirs(args.out_dir, exist_ok=True)

    # Single backup at the start; we restore on every error/exit
    backup = VOICES_JSON + ".seed_backup"
    shutil.copy2(VOICES_JSON, backup)

    print(f"voices : {len(voice_names)}")
    print(f"seeds  : {seeds}")
    print(f"output : {os.path.abspath(args.out_dir)}/")
    print(f"total  : ~{len(voice_names) * len(seeds)} requests\n")

    summary: dict[str, list] = {}

    try:
        for i, voice_name in enumerate(voice_names, 1):
            text = pick_text(voice_name, args.text)
            print(f"[{i}/{len(voice_names)}] {voice_name}")
            print(f"  text: {text[:90]}{'...' if len(text) > 90 else ''}")
            results = run_voice(voice_name, voices, seeds, text, args.host, args.port, args.out_dir)
            summary[voice_name] = results
            ok = sum(1 for _, p, _ in results if p)
            print(f"  → {ok}/{len(seeds)} OK\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted — restoring voices.json...")
    finally:
        shutil.copy2(backup, VOICES_JSON)
        os.remove(backup)
        print("voices.json restored.")

    # Write summary
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Seed samples — {len(voice_names)} voices, seeds {seeds}\n")
        f.write("=" * 60 + "\n\n")
        for voice_name, results in summary.items():
            ok = [(s, path) for s, path, err in results if path]
            failed = [(s, err) for s, path, err in results if not path]
            f.write(f"{voice_name}:\n")
            for s, path in ok:
                f.write(f"  seed {s:5d}  {path}\n")
            for s, err in failed:
                f.write(f"  seed {s:5d}  FAILED: {err}\n")
            f.write("\n")

    total_ok = sum(1 for r in summary.values() for _, p, _ in r if p)
    total = sum(len(r) for r in summary.values())
    print(f"\nDone: {total_ok}/{total} samples generated.")
    print(f"Summary written to {summary_path}")
    print(f"\nListen to the WAV files, then add \"seed\": <number> to your chosen voices in voices.json")


if __name__ == "__main__":
    main()
