import os
import json
import torch
import sys

# Append the app directory to import faster_qwen3_tts
sys.path.append("/app/examples")
sys.path.insert(0, "/app")

from faster_qwen3_tts.model import FasterQwen3TTS

def main():
    config_file = "/config/voices.json"
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found.")
        return

    with open(config_file, "r") as f:
        voices = json.load(f)

    # Need to load the model
    # Read model path from QWEN_TTS_MODEL or default
    model_path = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"Loading model {model_path} for embedding extraction...")
    
    tts_model = FasterQwen3TTS.from_pretrained(
        model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully.")

    updates_made = False

    for voice_id, entry in voices.items():
        spk_emb_path = entry.get("speaker_embeddings") or entry.get("speaker embeddings")
        
        # If there's already a valid path and it exists, skip
        if spk_emb_path and os.path.exists(spk_emb_path):
            print(f"Skipping {voice_id}, embedding already exists at {spk_emb_path}")
            continue
            
        ref_audio = entry.get("ref_audio")
        ref_text = entry.get("ref_text", "")
        
        if not ref_audio or not os.path.exists(ref_audio):
            print(f"Skipping {voice_id}, ref_audio not found: {ref_audio}")
            continue
            
        print(f"Extracting embeddings for {voice_id}...")
        try:
            prompt_items = tts_model.model.create_voice_clone_prompt(
                ref_audio, [ref_text]
            )
            vcp = tts_model.model._prompt_items_to_voice_clone_prompt(prompt_items)
            
            # Save the .pt file in /config/speakers
            pt_path = f"/config/speakers/{voice_id}.pt"
            torch.save(vcp, pt_path)
            print(f"Saved {pt_path}")
            
            # Update voices.json entry
            entry["speaker_embeddings"] = pt_path
            
            # Remove the legacy "speaker embeddings" with space if it exists
            if "speaker embeddings" in entry:
                del entry["speaker embeddings"]
                
            updates_made = True
        except Exception as e:
            print(f"Error extracting embedding for {voice_id}: {e}")

    if updates_made:
        with open(config_file, "w") as f:
            json.dump(voices, f, indent=2, ensure_ascii=False)
        print("Updated voices.json with new speaker_embeddings paths.")
    else:
        print("No new embeddings extracted.")

if __name__ == "__main__":
    main()
