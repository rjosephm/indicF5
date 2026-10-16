import torch
import soundfile as sf
from transformers import AutoModel

print("Loading IndicF5...")
model = AutoModel.from_pretrained(
    "ai4bharat/IndicF5",
    trust_remote_code=True
)

reference_audio = r"C:\AI\IndicF5\reference\malayalam_ref.wav"
reference_text = "ഇനി ഗ്രോത് ഹോർമോണിന് എന്തെങ്കിലും പ്രബലമായ ഒരു പ്രഭാവം ബ്രോയ്ലർ കോഴികളുടെ വളർച്ചയിൽ ഉണ്ടാക്കണം"
test_text = "നമസ്കാരം, ഇന്ന് നമുക്ക് ഒരു പുതിയ കഥ തുടങ്ങാം."

print("Generating Malayalam speech...")
with torch.inference_mode():
    audio = model(
        text=test_text,
        ref_audio_path=reference_audio,
        ref_text=reference_text
    )

sf.write(r"C:\AI\IndicF5\test_malayalam.wav", audio, 24000)

print("Saved: C:\\AI\\IndicF5\\test_malayalam.wav")
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
print("GPU memory allocated:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")
