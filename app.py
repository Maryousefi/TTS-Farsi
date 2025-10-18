# app.py
"""
FastAPI app for Persian TTS using facebook/mms-tts-fas (VITS)
This file consolidates the notebook's cells:
 - dependency imports & checks
 - Persian cleaners and phoneme helpers
 - audio enhancement function
 - model loading
 - simple web UI + synth API
"""

import io
import os
import re
import traceback
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Transformers imports
from transformers import VitsModel, AutoTokenizer

# -----------------------
# Configuration
# -----------------------
MODEL_ID = os.environ.get("MODEL_ID", "facebook/mms-tts-fas")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Persian cleaners (from notebook)
# -----------------------
AR2FA_MAP = {
    "ك": "ک", "ي": "ی", "ۀ": "ه", "ة": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
    "ئ": "ی", "ء": "", "ى": "ی", "ۃ": "ه",
}
FARSI_DIGITS = {'۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','۶':'6','۷':'7','۸':'8','۹':'9'}
ARABIC_DIGITS = {'٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'}

def normalize_digits(text: str) -> str:
    for fa, en in FARSI_DIGITS.items():
        text = text.replace(fa, en)
    for ar, en in ARABIC_DIGITS.items():
        text = text.replace(ar, en)
    return text

def normalize_persian_manual(text: str) -> str:
    for ar, fa in AR2FA_MAP.items():
        text = text.replace(ar, fa)
    text = normalize_digits(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def persian_cleaners(text: str) -> str:
    if not text or not text.strip():
        return " "
    text = normalize_persian_manual(text)
    return text

# -----------------------
# Simple Persian phoneme / vowel enhancer (kept conservative)
# -----------------------
PERSIAN_VOWELS_MAP = {
    # small mapping to normalize some sequences, not a full transliteration
    "می‌": "می ",
    "نمی‌": "نمی ",
    "‌ی": " ی",
}

def persian_phoneme_enhancer(text: str) -> str:
    # conservative enhancer: apply simple replacements that help spacing
    out = text
    for k, v in PERSIAN_VOWELS_MAP.items():
        out = out.replace(k, v)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# -----------------------
# Audio enhancement (same as notebook)
# -----------------------
from scipy import signal

def enhance_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Enhance audio: remove DC, high-pass, gentle compression, normalize."""
    audio = audio - np.mean(audio)
    # high-pass filter
    sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
    audio = signal.sosfilt(sos, audio)
    audio = np.tanh(audio * 1.2)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio

# -----------------------
# Model loading (runs at startup)
# -----------------------
print(f"Starting app. Device: {DEVICE}. Loading model: {MODEL_ID}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    SAMPLING_RATE = getattr(model.config, "sampling_rate", 16000)
    print(f"Model loaded on {DEVICE}. Sampling rate = {SAMPLING_RATE}")
except Exception as e:
    print("Failed to load model:", e)
    traceback.print_exc()
    raise

# -----------------------
# FastAPI app + templates
# -----------------------
app = FastAPI(title="Persian TTS Web Service")

# Static and templates: create directories if missing
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
if not os.path.exists("templates"):
    os.makedirs("templates", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Simple page with a textarea and a submit button.
    """
    return templates.TemplateResponse("index.html", {"request": request, "device": str(DEVICE)})

class SynthesizeRequest(BaseModel):
    text: str
    apply_phoneme_enhancer: Optional[bool] = False

@app.post("/synthesize")
async def synthesize(form_text: Optional[str] = Form(None), apply_phoneme_enhancer: Optional[bool] = Form(False)):
    """
    Endpoint to synthesize text and return a WAV audio stream.
    form_text: text to synthesize (form field name from index.html)
    apply_phoneme_enhancer: optional checkbox
    """
    text = (form_text or "").strip()
    if not text:
        return RedirectResponse(url="/")

    # Prepare text
    try:
        cleaned = persian_cleaners(text)
        if apply_phoneme_enhancer:
            cleaned = persian_phoneme_enhancer(cleaned)

        inputs = tokenizer(cleaned, return_tensors="pt").to(DEVICE)

        # Generate waveform (inference with torch.inference_mode)
        with torch.inference_mode():
            output = model(**inputs).waveform[0].cpu().numpy()

        # Post process
        audio = enhance_audio(output, SAMPLING_RATE)

        # write to bytes buffer as 16-bit PCM WAV
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLING_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)

        filename = "tts_output.wav"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return StreamingResponse(buf, media_type="audio/wav", headers=headers)

    except Exception as e:
        traceback.print_exc()
        return HTMLResponse(
            content=f"<h3>Failed to synthesize: {str(e)}</h3><pre>{traceback.format_exc()}</pre>",
            status_code=500,
        )

# simple health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_ID}
