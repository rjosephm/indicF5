import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

BASE = Path(r"C:\AI\IndicF5")
PROFILE_DIR = Path(
    r"C:\MoneyPrinterTurbo -facelesee ai\storage\indicf5_profiles"
)
FEMALE_CSV = BASE / "femaleSample" / "mapping.csv"
MALE_CSV = BASE / "maleSample" / "mapping.csv"
REPORT_CSV = BASE / "profile_creation_report.csv"

def load_mapping(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")

    result = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"filename", "speaker_id", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns: filename, speaker_id, text"
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            transcript = row.get("text") or ""
            if filename and transcript.strip():
                result[filename.casefold()] = {
                    "filename": filename,
                    "speaker_id": (row.get("speaker_id") or "").strip(),
                    "text": transcript,
                }
    return result

def main():
    if not PROFILE_DIR.is_dir():
        raise FileNotFoundError(f"Profile directory not found: {PROFILE_DIR}")

    female = load_mapping(FEMALE_CSV)
    male = load_mapping(MALE_CSV)

    # Optional report helps connect each profile ID to its original WAV.
    report = {}
    if REPORT_CSV.exists():
        with REPORT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pid = (row.get("profile_id") or "").strip()
                wav = (row.get("audio_filename") or "").strip()
                if pid and wav:
                    report[pid.casefold()] = wav

    profiles = sorted(PROFILE_DIR.glob("*.json"))
    changes = []
    unmatched = []
    errors = []

    for path in profiles:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            pid = str(data.get("id") or data.get("name") or path.stem).strip()
            pid_lower = pid.casefold()

            if pid_lower.startswith("ml-female-"):
                mapping = female
                gender = "FEMALE"
            elif pid_lower.startswith("ml-male-"):
                mapping = male
                gender = "MALE"
            else:
                continue

            candidates = []
            report_wav = report.get(pid_lower)
            if report_wav:
                candidates.append(Path(report_wav).name)

            audio_path = str(data.get("reference_audio") or "")
            if audio_path:
                # Works with both Windows and slash-separated paths.
                candidates.append(audio_path.replace("/", "\\").split("\\")[-1])

            match = None
            for candidate in candidates:
                match = mapping.get(candidate.casefold())
                if match:
                    break

            if not match:
                unmatched.append(
                    f"{path.name}: no exact WAV filename match "
                    f"(report={report_wav!r}, audio={audio_path!r})"
                )
                continue

            old_text = data.get("reference_text", "")
            if old_text == match["text"]:
                changes.append((path, gender, match, "ALREADY CORRECT", data))
            else:
                changes.append((path, gender, match, "UPDATED", data))

        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    print(f"Profile JSON files found: {len(profiles)}")
    print(f"Female mapping entries: {len(female)}")
    print(f"Male mapping entries: {len(male)}")
    print(f"Profiles matched: {len(changes)}")
    print(f"Unmatched profiles: {len(unmatched)}")
    print(f"Read errors: {len(errors)}")

    if not changes:
        print("No profiles were changed.")
        for item in unmatched:
            print("UNMATCHED:", item)
        for item in errors:
            print("ERROR:", item)
        return

    # Back up all profile JSON files before any modification.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROFILE_DIR / f"_backup_before_transcript_fix_{timestamp}"
    backup_dir.mkdir()
    for path in profiles:
        shutil.copy2(path, backup_dir / path.name)

    updated = 0
    unchanged = 0
    audit_rows = []

    for path, gender, match, status, data in changes:
        if status == "UPDATED":
            data["reference_text"] = match["text"]
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            updated += 1
        else:
            unchanged += 1

        audit_rows.append({
            "profile_id": data.get("id") or data.get("name") or path.stem,
            "gender": gender,
            "matched_filename": match["filename"],
            "matched_speaker_id": match["speaker_id"],
            "status": status,
            "reference_text": match["text"],
        })

    audit_path = BASE / "profile_transcript_correction_report.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "profile_id", "gender", "matched_filename",
                "matched_speaker_id", "status", "reference_text"
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print("\nCORRECTION COMPLETE")
    print(f"Updated: {updated}")
    print(f"Already correct: {unchanged}")
    print(f"Backup folder: {backup_dir}")
    print(f"Audit report: {audit_path}")

    for item in unmatched:
        print("UNMATCHED:", item)
    for item in errors:
        print("ERROR:", item)

if __name__ == "__main__":
    main()
