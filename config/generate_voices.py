#generate_voices.py

import os
import json
import re
import subprocess

output_file = "/config/voices.json"
converted_dir = "/config/converted"

# .m4a is converted to WAV on the fly because soundfile doesn't support AAC
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".m4a")
SKIP_DIRS = {"originals_backup", "xtts_multi_voice_sets", "txt"}

# (base_dir_on_container, container_path_prefix)
# /config/speakers — legacy location, writable
# /voices          — new external mount, read-only
SCAN_DIRS = [
    "/config/speakers",
    "/voices",
]

# Load existing voices.json so manually added fields (temperature, top_k,
# top_p, chunk_size overrides, etc.) survive a container restart.
_existing = {}
if os.path.exists(output_file):
    try:
        with open(output_file, encoding="utf-8") as _f:
            _existing = json.load(_f)
    except (json.JSONDecodeError, OSError):
        pass

voices = {}


def detect_language(base_name):
    if base_name.startswith("EN_") or base_name.startswith("basic_ref_en"):
        return "English"
    if base_name.startswith("DE_"):
        return "German"
    if base_name.startswith("basic_ref_zh"):
        return "Chinese"
    return "Auto"




def make_voice_id(base_dir, root, base_name):
    rel = os.path.relpath(root, base_dir)
    parts = [] if rel == "." else rel.split(os.sep)
    parts.append(base_name)
    raw = "_".join(parts)
    return re.sub(r"[^\w\-]", "_", raw)


def convert_m4a(src_path, voice_id):
    """Convert M4A to WAV in /config/converted/. Returns the WAV path."""
    os.makedirs(converted_dir, exist_ok=True)
    dst_path = os.path.join(converted_dir, f"{voice_id}.wav")
    if not os.path.exists(dst_path):
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "24000", "-ac", "1", dst_path],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"  ✗ ffmpeg failed for {src_path}: {result.stderr.decode()[:200]}")
            return None
        print(f"  Converted: {os.path.basename(src_path)} → {dst_path}")
    return dst_path


for scan_dir in SCAN_DIRS:
    if not os.path.exists(scan_dir):
        print(f"Skipping {scan_dir} (not mounted)")
        continue

    for root, dirs, files in os.walk(scan_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        for filename in sorted(files):
            if not filename.lower().endswith(AUDIO_EXTS):
                continue

            base_name = os.path.splitext(filename)[0]
            audio_path = os.path.join(root, filename)
            voice_id = make_voice_id(scan_dir, root, base_name)

            if filename.lower().endswith(".m4a"):
                audio_path = convert_m4a(audio_path, voice_id)
                if audio_path is None:
                    continue

            entry = {
                "ref_audio": audio_path,
                "language": detect_language(base_name),
                "chunk_size": 4,
            }

            ref_txt = os.path.join(root, f"{base_name}.reference.txt")
            txt = os.path.join(root, f"{base_name}.txt")
            if os.path.exists(ref_txt):
                with open(ref_txt, encoding="utf-8") as f:
                    entry["ref_text"] = f.read().strip()
            elif os.path.exists(txt):
                with open(txt, encoding="utf-8") as f:
                    entry["ref_text"] = f.read().strip()

            # Preserve any user-added fields from the previous voices.json
            # (temperature, top_k, top_p, chunk_size overrides, etc.)
            if voice_id in _existing:
                for key, val in _existing[voice_id].items():
                    if key not in entry:
                        entry[key] = val

            voices[voice_id] = entry

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(voices, f, indent=2, ensure_ascii=False)

print(f"Success! Generated voices.json with {len(voices)} mapped voices.")
