from collections import defaultdict
import csv
import os
import shutil
import zipfile

# Define paths
root_dir = r"C:\AI\IndicF5"
zip_path = os.path.join(root_dir, "ml_in_male.zip")
extract_dir = os.path.join(root_dir, "ml_in_male_extracted")
output_dir = os.path.join(root_dir, "maleSample")

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# 1. Extract the zip file if it hasn't been extracted yet
if os.path.exists(zip_path):
  print(f"Extracting {zip_path}...")
  with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
else:
  print(f"Zip file not found at {zip_path}. Looking for extracted folders...")
  extract_dir = root_dir

# 2. Locate the line_index.tsv transcription file
tsv_path = None
for root, dirs, files in os.walk(extract_dir):
  if "line_index.tsv" in files:
    tsv_path = os.path.join(root, "line_index.tsv")
    break

print(f"Found metadata file: {tsv_path}")

# 3. Read the TSV and group clips by Speaker ID
speaker_clips = defaultdict(list)
if tsv_path and os.path.exists(tsv_path):
  with open(tsv_path, "r", encoding="utf-8") as f:
    for line in f:
      parts = line.strip().split("\t")
      if len(parts) >= 2:
        filename_rel = parts[0].strip()
        text = parts[1].strip()

        # Extract unique speaker ID (e.g., mlf_02879 from mlf_02879_01795762363)
        base_name = os.path.basename(filename_rel)
        speaker_id = base_name.rsplit("_", 1)[0]
        speaker_clips[speaker_id].append((filename_rel, text))

print(f"Total Unique Speakers Found: {len(speaker_clips)}")

# 4. Process each speaker: pick 1 clip, rename, and map
mapping_data = []
csv_path = os.path.join(output_dir, "mapping.csv")

for idx, (speaker_id, clips) in enumerate(speaker_clips.items(), start=1):
  if not clips:
    continue

  # Pick the first clip that has a valid text transcript
  selected_clip = clips[0]
  for c in clips:
    if c[1]:
      selected_clip = c
      break

  orig_path_rel, text = selected_clip
  base_target_filename = os.path.basename(orig_path_rel)

  # Ensure .wav extension is handled properly
  target_filename_wav = (
      base_target_filename
      if base_target_filename.endswith(".wav")
      else base_target_filename + ".wav"
  )

  # Search recursively inside extract_dir for the actual .wav file
  orig_file_path = None
  for r, ds, fs in os.walk(extract_dir):
    if target_filename_wav in fs:
      orig_file_path = os.path.join(r, target_filename_wav)
      break

  if orig_file_path and os.path.exists(orig_file_path):
    new_filename = f"M{idx:02d}.wav"
    dest_path = os.path.join(output_dir, new_filename)

    # Copy and rename file into femaleSample folder
    shutil.copy(orig_file_path, dest_path)

    mapping_data.append(
        {"filename": new_filename, "speaker_id": speaker_id, "text": text}
    )
    print(f"Mapped Speaker {speaker_id} -> {new_filename}")
  else:
    print(f"Warning: Audio file missing for {target_filename_wav}")

# 5. Write the mapping CSV
with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
  fieldnames = ["filename", "speaker_id", "text"]
  writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
  writer.writeheader()
  for row in mapping_data:
    writer.writerow(row)

print(
    f"\nSuccess! Generated {len(mapping_data)} sample files inside"
    f" '{output_dir}'"
)
print(f"CSV mapping saved to: {csv_path}")