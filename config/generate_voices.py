#generate_voices.py
"""
Scan the speakers directory for reference audio files and generate voices.json.

Runs inside the container at startup to map all available voice reference
audio files into a format the OpenAI-compatible TTS server understands.
"""

import os
import json

# Internal container paths
speaker_dir = "/config/speakers"
output_file = "/config/voices.json"

voices = {}

# Ensure the directory exists just in case
if os.path.exists(speaker_dir):
    for filename in os.listdir(speaker_dir):
        if filename.endswith((".wav", ".mp3")):
            base_name = os.path.splitext(filename)[0]

            # Determine language based on file prefixes
            lang = "Auto"
            if base_name.startswith("EN_") or base_name.startswith("basic_ref_en"):
                lang = "English"
            elif base_name.startswith("DE_"):
                lang = "German"
            elif base_name.startswith("basic_ref_zh"):
                lang = "Chinese"

        #  # Create a clean voice ID
        #    voice_id = base_name.lower()
        #    prefixes_to_strip = ["en_m_", "en_f_", "de_m_", "de_f_"]
        #    for prefix in prefixes_to_strip:
        #        if voice_id.startswith(prefix):
        #            voice_id = voice_id.replace(prefix, "", 1)
        #            break

            entry = {
                "ref_audio": f"/config/speakers/{filename}",
                "language": lang,
                "chunk_size": 4,
            }

            # Look for matching reference text files
            ref_txt_path = os.path.join(speaker_dir, f"{base_name}.reference.txt")
            txt_path = os.path.join(speaker_dir, f"{base_name}.txt")

            if os.path.exists(ref_txt_path):
                with open(ref_txt_path, 'r', encoding='utf-8') as f:
                    entry["ref_text"] = f.read().strip()
            elif os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    entry["ref_text"] = f.read().strip()

            voices[voice_id] = entry

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)

    print(f"Success! Generated voices.json with {len(voices)} mapped voices.")
else:
    print(f"Warning: Directory {speaker_dir} not found. Skipping voice generation.")
