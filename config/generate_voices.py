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

            # A .pt speaker embedding is a CACHE baked from a specific pairing of
            # reference audio + reference transcript. Re-recording or re-designing
            # a voice replaces those source files but leaves the old .pt sitting
            # there, and this script used to point at it unconditionally — so the
            # server kept cloning from an embedding whose internal audio tokens no
            # longer matched the transcript it was stored with. Confirmed live on
            # 73 voices: output ignored the requested text entirely, emitting short
            # unrelated filler ("Thank you.") or fragments of the OLD reference
            # transcript. Treat an embedding older than its sources as invalid so
            # the server recomputes it from the current audio.
            pt_path = os.path.join("/config/speakers", f"{voice_id}.pt")
            entry["speaker_embeddings"] = ""
            if os.path.exists(pt_path):
                try:
                    pt_mtime = os.path.getmtime(pt_path)
                    newest_source = os.path.getmtime(audio_path)
                    for src in (ref_txt, txt):
                        if os.path.exists(src):
                            newest_source = max(newest_source, os.path.getmtime(src))
                    # 1s slack absorbs filesystem timestamp granularity.
                    if newest_source > pt_mtime + 1:
                        print(f"  ⚠ stale embedding for {voice_id} — regenerating from current reference")
                        os.remove(pt_path)
                    else:
                        entry["speaker_embeddings"] = pt_path
                except OSError as exc:
                    print(f"  ✗ could not validate embedding for {voice_id}: {exc}")

            # Preserve any user-added fields from the previous voices.json
            # (temperature, top_k, top_p, chunk_size overrides, etc.)
            if voice_id in _existing:
                for key, val in _existing[voice_id].items():
                    if key not in entry:
                        entry[key] = val

            voices[voice_id] = entry

if voices != _existing:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)
    print(f"Success! Generated voices.json with {len(voices)} mapped voices.")
else:
    pass # No changes, do not update mtime


# ---------------------------------------------------------------------------
# Voice Design registry
# ---------------------------------------------------------------------------
# For the VoiceDesign model there is no reference audio and no embedding — a
# voice's identity IS its instruct prompt. Designed voices were never written
# into voicedesign_voices.json at all, so the server only knew its 8 bundled
# presets and silently substituted one of them for every custom voice
# (confirmed live: requests for a designed male German character were answered
# by the 'vd_british_male' preset). Mirror the app's designed voices here so
# the identity resolves to the prompt it was actually created from.
#
# This runs in the voice-clone container because that is the one with the
# voice library mounted; /config is shared with the VoiceDesign container,
# which hot-reloads this file.

VOICEDESIGN_OUTPUT = "/config/voicedesign_voices.json"
VOICEDESIGN_NOTE_PREFIX = "Voice Design:"

_LANG_BY_FLAG = {
    "EN": "English", "DE": "German", "FR": "French", "ES": "Spanish",
    "IT": "Italian", "PT": "Portuguese", "NL": "Dutch", "PL": "Polish",
    "ZH": "Chinese", "JA": "Japanese", "KO": "Korean",
}


def _build_voicedesign_registry():
    try:
        with open(VOICEDESIGN_OUTPUT, encoding="utf-8") as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        existing = {}

    # Keep the bundled vd_* presets; rebuild every app-managed entry.
    registry = {k: v for k, v in existing.items() if k.startswith("vd_")}
    added = 0

    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for filename in sorted(files):
                if not filename.endswith(".meta.json"):
                    continue
                try:
                    with open(os.path.join(root, filename), encoding="utf-8") as f:
                        meta = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if meta.get("origin") != "designed":
                    continue

                # Prefer the full prompt. `note` is a display summary the app
                # clips to 240 characters, which for older voices is the only
                # copy that survives — usable (it still carries gender, accent
                # and timbre) but missing the tail of the description.
                instruct = str(meta.get("voice_design_prompt") or "").strip()
                if not instruct:
                    note = str(meta.get("note") or "").strip()
                    if not note.startswith(VOICEDESIGN_NOTE_PREFIX):
                        continue
                    instruct = note[len(VOICEDESIGN_NOTE_PREFIX):].strip()
                if not instruct:
                    continue

                base_name = filename[: -len(".meta.json")]
                voice_id = make_voice_id(scan_dir, root, base_name)
                flag = str(meta.get("flag") or "").upper()
                entry = {
                    "instruct": instruct,
                    "language": _LANG_BY_FLAG.get(flag) or detect_language(base_name),
                }
                # Deliberately no "seed": generate_voice_design() takes no seed
                # parameter, so a designed voice cannot be pinned that way.
                # Consistency across takes comes from low temperature/top_p,
                # which the server reads from the entry when present. Preserve
                # any values a previous run or the user set by hand.
                for key in ("temperature", "top_p", "top_k"):
                    prev = existing.get(voice_id, {}).get(key)
                    if prev is not None:
                        entry[key] = prev
                registry[voice_id] = entry
                added += 1

    if registry != existing:
        with open(VOICEDESIGN_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        print(f"Success! Generated voicedesign_voices.json with {added} designed voices.")


_build_voicedesign_registry()

