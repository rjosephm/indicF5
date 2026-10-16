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
