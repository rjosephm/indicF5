import os
import time
import io
import json
import librosa
import soundfile as sf
from datasets import load_dataset, Audio

print("Loading Malayalam dataset stream from IndicVoices-R...")
dataset = load_dataset("ai4bharat/indicvoices_r", "Malayalam", split="train", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False))

target_male_count = 10
target_female_count = 10

collected_males = {}
collected_females = {}

print(f"Scanning for {target_male_count} masculine males and {target_female_count} girlish females...")

for sample in dataset:
    gender = str(sample.get("gender", "")).strip().lower()
    age = str(sample.get("age", sample.get("age_group", "unknown"))).strip().lower()
    speaker_id = sample.get("speaker_id", "unknown")
    audio_info = sample.get("audio")
    text = sample.get("text", sample.get("sentence", ""))
    
    pitch_mean = sample.get("utterance_pitch_mean", 0.0)
    try:
        pitch_mean = float(pitch_mean)
    except (TypeError, ValueError):
        pitch_mean = 0.0

    if not audio_info or not text:
        continue

    if gender in ["male", "m"] and len(collected_males) < target_male_count:
        if speaker_id not in collected_males:
            if pitch_mean == 0.0 or pitch_mean < 130.0:
                collected_males[speaker_id] = {"sample": sample, "age": age, "pitch": pitch_mean}
                print(f"[Found Masculine Male {len(collected_males)}/{target_male_count}] ID: {speaker_id}, Pitch: {pitch_mean:.1f}Hz")

    elif gender in ["female", "f"] and len(collected_females) < target_female_count:
        if speaker_id not in collected_females:
            if pitch_mean == 0.0 or pitch_mean > 210.0:
                collected_females[speaker_id] = {"sample": sample, "age": age, "pitch": pitch_mean}
                print(f"[Found Girlish Female {len(collected_females)}/{target_female_count}] ID: {speaker_id}, Pitch: {pitch_mean:.1f}Hz")

    if len(collected_males) >= target_male_count and len(collected_females) >= target_female_count:
        break

def download_with_retry(audio_info, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            audio_bytes = audio_info.get("bytes")
            if audio_bytes:
                return sf.read(io.BytesIO(audio_bytes))
            else:
                path = audio_info.get("path")
                return librosa.load(path, sr=None)
        except Exception as e:
            print(f"Connection dropped ({e}). Retrying {attempt}/{max_retries}...")
            time.sleep(2 * attempt)
    raise RuntimeError("Failed to download audio chunk after multiple retries.")

print("\nDownloading audio files and generating manifest transcripts...")
manifest = []

# Process Males
for idx, (spk_id, data) in enumerate(collected_males.items(), start=1):
    filename = f"malayalam_masculine_male_{idx}.wav"
    s = data["sample"]
    transcript = s.get('text', s.get('sentence', ''))
    
    if not os.path.exists(filename):
        audio_array, sr = download_with_retry(s["audio"])
        trimmed = audio_array[:7 * sr]
        sf.write(filename, trimmed, sr)
        print(f"Saved: {filename}")
    else:
        print(f"Skipping (already exists): {filename}")

    manifest.append({
        "file": filename,
        "type": "masculine_male",
        "pitch_hz": round(data['pitch'], 1),
        "ref_text": transcript
    })

# Process Females
for idx, (spk_id, data) in enumerate(collected_females.items(), start=1):
    filename = f"malayalam_girlish_female_{idx}.wav"
    s = data["sample"]
    transcript = s.get('text', s.get('sentence', ''))
    
    if not os.path.exists(filename):
        audio_array, sr = download_with_retry(s["audio"])
        trimmed = audio_array[:7 * sr]
        sf.write(filename, trimmed, sr)
        print(f"Saved: {filename}")
    else:
        print(f"Skipping (already exists): {filename}")

    manifest.append({
        "file": filename,
        "type": "girlish_female",
        "pitch_hz": round(data['pitch'], 1),
        "ref_text": transcript
    })

# Save manifest to JSON file
manifest_path = "voice_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=4)

print(f"\n--- SUCCESS! Manifest saved to {manifest_path} ---")
print(f"Total files ready: {len(manifest)} ({target_male_count} males, {target_female_count} females)")
