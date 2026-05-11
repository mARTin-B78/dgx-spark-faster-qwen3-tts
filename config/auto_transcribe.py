import os
import requests
import json

# Host-side paths (this script runs on the host, not inside the container)
SCAN_DIRS = [
    "/home/sparky/Docker/faster-qwen3-tts/config/speakers",
    "/home/sparky/Projekte/TTS_Voices/speakers",
]
AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".m4a")
SKIP_DIRS = {"originals_backup", "xtts_multi_voice_sets", "txt"}

whisper_api_url = "http://localhost:8010/v1/audio/transcriptions"

for scan_dir in SCAN_DIRS:
    if not os.path.exists(scan_dir):
        print(f"Skipping {scan_dir} (not found)")
        continue

    print(f"\nScanning {scan_dir} for missing transcripts...")

    for root, dirs, files in os.walk(scan_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        for filename in sorted(files):
            if not filename.lower().endswith(AUDIO_EXTS):
                continue

            base_name = os.path.splitext(filename)[0]
            ref_txt_path = os.path.join(root, f"{base_name}.reference.txt")
            audio_path = os.path.join(root, filename)

            if os.path.exists(ref_txt_path):
                continue

            print(f"Transcribing: {os.path.relpath(audio_path, scan_dir)}")
            try:
                with open(audio_path, "rb") as audio_file:
                    response = requests.post(
                        whisper_api_url,
                        files={"file": (filename, audio_file)},
                        data={"model": "large-v3", "response_format": "text"},
                    )

                if response.status_code == 200:
                    transcript = response.text.strip()
                    if transcript.startswith("{"):
                        try:
                            transcript = json.loads(transcript).get("text", transcript).strip()
                        except json.JSONDecodeError:
                            pass

                    with open(ref_txt_path, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    print(f"  ✓ {transcript[:80]}")
                else:
                    print(f"  ✗ API error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                print(f"  ✗ Cannot reach Whisper API at {whisper_api_url}")
                raise SystemExit(1)
            except Exception as e:
                print(f"  ✗ Error on {filename}: {e}")

print("\nBatch transcription complete.")
