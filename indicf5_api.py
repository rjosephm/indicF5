import os
import threading
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModel

BASE = Path(r"C:\AI\IndicF5")
DEFAULT_AUDIO = BASE / "reference" / "malayalam_ref.wav"
MODEL_ID = "ai4bharat/IndicF5"

app = FastAPI(title="Local IndicF5 Malayalam TTS API")
model = None
generation_lock = threading.Lock()


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    reference_text: str = Field(min_length=1, max_length=2000)
    reference_audio: str = str(DEFAULT_AUDIO)


@app.on_event("startup")
def load_model():
    global model
    print("Loading IndicF5 model...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    print("IndicF5 loaded successfully.")


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0)
        if torch.cuda.is_available() else None
    }


def trim_leading_noise(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Trim sustained low-level noise before speech, retaining a short onset buffer.

    This is a conservative amplitude-based heuristic, not a general denoiser.
    It only searches the first two seconds and leaves 60 ms before the detected onset.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    frame_size = max(1, int(sample_rate * 0.020))  # 20 ms frames
    search_samples = min(audio.size, int(sample_rate * 2.0))
    segment = audio[:search_samples]
    frame_count = (segment.size + frame_size - 1) // frame_size
    rms = np.zeros(frame_count, dtype=np.float32)

    for i in range(frame_count):
        frame = segment[i * frame_size:(i + 1) * frame_size]
        if frame.size:
            rms[i] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))

    if rms.size < 4:
        return audio

    threshold = max(0.0025, float(rms.max()) * 0.08)
    active = rms >= threshold
    onset_frame = None
    for i in range(max(0, len(active) - 2)):
        if active[i] and active[i + 1] and active[i + 2]:
            onset_frame = i
            break

    if onset_frame is None:
        return audio

    onset_sample = onset_frame * frame_size
    keep_from = max(0, onset_sample - int(sample_rate * 0.060))
    # Avoid trimming tiny amounts that are unlikely to be meaningful.
    if keep_from < int(sample_rate * 0.020):
        return audio
    return audio[keep_from:]


@app.post("/generate")
def generate(req: SpeechRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    audio_path = Path(req.reference_audio).resolve()
    if not audio_path.is_file():
        raise HTTPException(status_code=400, detail="Reference audio not found")

    # Restrict reference files to the IndicF5 reference directory.
    allowed_dir = (BASE / "reference").resolve()
    if audio_path.parent != allowed_dir:
        raise HTTPException(status_code=400, detail="Reference audio must be in the reference directory")

    try:
        with generation_lock:
            with torch.inference_mode():
                audio = model(
                    text=req.text,
                    ref_audio_path=str(audio_path),
                    ref_text=req.reference_text
                )

        audio = np.asarray(audio).squeeze()
        if audio.size == 0 or audio.ndim != 1:
            raise RuntimeError("Model returned invalid audio")

        # Remove sustained low-level noise at the start of generated speech.
        audio = trim_leading_noise(audio, sample_rate=24000)

        output_dir = BASE / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(
            suffix=".wav", prefix="indicf5_", dir=output_dir
        )
        os.close(fd)
        sf.write(filename, audio, 24000, subtype="PCM_16")

        return FileResponse(
            filename,
            media_type="audio/wav",
            filename="indicf5_speech.wav",
            background=None
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
