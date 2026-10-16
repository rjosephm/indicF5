#!/usr/bin/env python3
"""
Add the new WAV samples listed in voice_manifest.json to the existing
IndicF5 + MoneyPrinterTurbo profile setup without overwriting old profiles.

Default setup:
  IndicF5 folder: C:\\AI\\IndicF5
  MoneyPrinterTurbo folder: C:\\MoneyPrinterTurbo -facelesee ai
  Manifest: C:\\AI\\IndicF5\\voice_manifest.json

New IDs:
  Female samples: a1-ml-female-23, a1-ml-female-24, ...
  Male samples:   a1-ml-male-19,   a1-ml-male-20, ...

The script copies source WAVs into C:\\AI\\IndicF5\\reference\\added_samples
and creates profile JSONs in MoneyPrinterTurbo's storage\\indicf5_profiles.
It never deletes profiles or overwrites an existing profile/WAV. If a target
already exists, it picks the next unused profile number or a unique WAV name.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

# Change these only if your folders are different.
INDICF5_DIR = Path(r"C:\AI\IndicF5")
MPT_DIR = Path(r"C:\MoneyPrinterTurbo -facelesee ai")
MANIFEST_PATH = INDICF5_DIR / "voice_manifest.json"
PROFILE_DIR = MPT_DIR / "storage" / "indicf5_profiles"
REFERENCE_DIR = INDICF5_DIR / "reference" / "added_samples"
API_URL = "http://127.0.0.1:8765"

# The requested example starts female additions at 23. Existing unprefixed
# ml-female-23 remains untouched because these IDs have the "a1-" prefix.
FEMALE_START = 23
MALE_START = 19


def norm_gender(item: dict) -> str | None:
    """Return female or male based on manifest type or filename."""
    text = f"{item.get('type', '')} {item.get('file', '')}".lower()
    if "female" in text or re.search(r"\bf\d+\.wav\b", text):
        return "female"
    if "masculine_male" in text or "male" in text or re.search(r"\bm\d+\.wav\b", text):
        return "male"
    return None


def locate_audio(filename: str) -> Path | None:
    """Find a manifest WAV beside the manifest or under the IndicF5 folder."""
    raw = Path(filename)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([MANIFEST_PATH.parent / raw, INDICF5_DIR / raw])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Fallback: search below the IndicF5 directory by basename.
    matches = list(INDICF5_DIR.rglob(raw.name))
    return matches[0].resolve() if matches else None


def next_free_id(gender: str, start: int) -> str:
    """Use requested numbering, skipping any profile IDs already present."""
    number = start
    while True:
        profile_id = f"a1-ml-{gender}-{number:02d}"
        if not (PROFILE_DIR / f"{profile_id}.json").exists():
            return profile_id
        number += 1


def unique_audio_destination(source: Path, profile_id: str) -> Path:
    dest = REFERENCE_DIR / f"{profile_id}_{source.name}"
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    n = 2
    while (REFERENCE_DIR / f"{stem}_{n}{suffix}").exists():
        n += 1
    return REFERENCE_DIR / f"{stem}_{n}{suffix}"


def main() -> int:
    if not MANIFEST_PATH.is_file():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        print("Edit INDICF5_DIR / MANIFEST_PATH at the top of this script.")
        return 1

    if not PROFILE_DIR.parent.exists():
        print(f"ERROR: MoneyPrinterTurbo folder not found: {MPT_DIR}")
        print("Edit MPT_DIR at the top of this script.")
        return 1

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"ERROR: Could not read manifest JSON: {exc}")
        return 1

    if isinstance(manifest, dict):
        # Accept common wrappers such as {"voices": [...]}.
        for key in ("voices", "samples", "items", "profiles"):
            if isinstance(manifest.get(key), list):
                manifest = manifest[key]
                break

    if not isinstance(manifest, list):
        print("ERROR: voice_manifest.json must contain a JSON list of sample objects.")
        return 1

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MPT_DIR / "storage" / "indicf5_added_voice_report.csv"

    report_rows = []
    created = skipped = errors = 0
    counters = {"female": FEMALE_START, "male": MALE_START}

    for item in manifest:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("file", "")).strip()
        ref_text = str(item.get("ref_text", item.get("text", ""))).strip()
        gender = norm_gender(item)

        if not filename:
            continue
        if gender is None:
            report_rows.append({
                "source_file": filename, "gender": "", "profile_id": "",
                "status": "SKIPPED: could not infer gender from type/filename",
                "profile_json": "", "copied_audio": ""
            })
            skipped += 1
            continue
        if not ref_text:
            report_rows.append({
                "source_file": filename, "gender": gender, "profile_id": "",
                "status": "SKIPPED: ref_text/text is empty",
                "profile_json": "", "copied_audio": ""
            })
            skipped += 1
            continue

        source = locate_audio(filename)
        if source is None:
            report_rows.append({
                "source_file": filename, "gender": gender, "profile_id": "",
                "status": "ERROR: WAV not found under IndicF5 folder",
                "profile_json": "", "copied_audio": ""
            })
            errors += 1
            continue

        profile_id = next_free_id(gender, counters[gender])
        counters[gender] = int(profile_id.rsplit("-", 1)[1]) + 1
        profile_path = PROFILE_DIR / f"{profile_id}.json"
        dest_audio = unique_audio_destination(source, profile_id)

        try:
            shutil.copy2(source, dest_audio)
            profile = {
                "id": profile_id,
                "name": profile_id,
                "reference_audio": str(dest_audio.resolve()),
                "reference_text": ref_text,
                "api_url": API_URL,
            }
            # Exclusive creation is an extra safeguard against overwrites.
            with profile_path.open("x", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
                f.write("\n")

            report_rows.append({
                "source_file": str(source), "gender": gender, "profile_id": profile_id,
                "status": "CREATED", "profile_json": str(profile_path),
                "copied_audio": str(dest_audio)
            })
            created += 1
            print(f"CREATED  {profile_id}  <-  {source.name}")
        except FileExistsError:
            # Should be rare; never overwrite. Let the user rerun safely.
            report_rows.append({
                "source_file": str(source), "gender": gender, "profile_id": profile_id,
                "status": "SKIPPED: profile already exists; nothing overwritten",
                "profile_json": str(profile_path), "copied_audio": str(dest_audio)
            })
            skipped += 1
        except Exception as exc:
            errors += 1
            report_rows.append({
                "source_file": str(source), "gender": gender, "profile_id": profile_id,
                "status": f"ERROR: {exc}", "profile_json": str(profile_path),
                "copied_audio": str(dest_audio)
            })
            print(f"ERROR    {filename}: {exc}", file=sys.stderr)

    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["source_file", "gender", "profile_id", "status", "profile_json", "copied_audio"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)

    print("\nFinished.")
    print(f"Created: {created} | Skipped: {skipped} | Errors: {errors}")
    print(f"Profile folder: {PROFILE_DIR}")
    print(f"Copied WAV folder: {REFERENCE_DIR}")
    print(f"Report: {report_path}")
    print("\nRestart MoneyPrinterTurbo (or refresh its voice list) to see the new profiles.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
