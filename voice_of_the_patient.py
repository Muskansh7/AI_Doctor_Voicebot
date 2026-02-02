import os
from datetime import datetime
import tempfile

import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

from google import genai
from google.genai import types


# ---------- FFmpeg CONFIG ----------
AudioSegment.converter = r"C:\Users\muska\OneDrive\Desktop\AI_Doctor_Voicebot\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\muska\OneDrive\Desktop\AI_Doctor_Voicebot\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"


# ---------- GEMINI SETUP ----------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY missing")

client = genai.Client(api_key=api_key)


# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = (
    "You are a medical assistant. "
    "Provide only general observations and informational responses. "
    "Do not provide diagnosis or medical advice."
)


# ---------- STORAGE ----------
BASE_DIR = "recordings"
PATIENT_DIR = os.path.join(BASE_DIR, "patient_inputs")
DOCTOR_DIR = os.path.join(BASE_DIR, "doctor_responses")

os.makedirs(PATIENT_DIR, exist_ok=True)
os.makedirs(DOCTOR_DIR, exist_ok=True)


# ---------- GEMINI TEXT PART ----------
def text_part(text):
    return types.Part.from_text(text)


# ---------- GEMINI IMAGE PART ----------
def image_part(path):
    with open(path, "rb") as f:
        return types.Part.from_bytes(
            data=f.read(),
            mime_type="image/jpeg"
        )


# ---------- SPEECH → TEXT ----------
def speech_to_text(audio_np, sample_rate):
    r = sr.Recognizer()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sample_rate, audio_np)
        with sr.AudioFile(f.name) as source:
            audio_data = r.record(source)

    os.remove(f.name)
    return r.recognize_google(audio_data)


# ---------- DOCTOR VOICE ----------
def generate_doctor_voice(text):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(DOCTOR_DIR, f"doctor_response_{ts}.mp3")

    gTTS(text=text, lang="en", slow=False).save(audio_path)
    return audio_path


# ---------- MAIN PIPELINE FOR GRADIO ----------
def patient_pipeline(mic_audio, image_path):
    if mic_audio is None:
        return "Please record your voice.", None, None

    sample_rate, audio = mic_audio
    audio = audio.astype(np.int16)

    # Save patient audio
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_wav = os.path.join(PATIENT_DIR, f"patient_{ts}.wav")
    patient_mp3 = os.path.join(PATIENT_DIR, f"patient_{ts}.mp3")

    wav.write(patient_wav, sample_rate, audio)
    AudioSegment.from_wav(patient_wav).export(patient_mp3, format="mp3")
    os.remove(patient_wav)

    # Noise reduction
    cleaned = nr.reduce_noise(y=audio.flatten().astype(np.float32), sr=sample_rate)
    cleaned = cleaned.astype(np.int16)

    try:
        patient_text = speech_to_text(cleaned, sample_rate)
    except Exception:
        return "Could not understand speech.", None, None

    # Gemini request
    contents = [
        types.Content(
            role="user",
            parts=[
                text_part(SYSTEM_PROMPT),
                text_part(patient_text)
            ]
        )
    ]

    if image_path:
        try:
            contents[0].parts.append(image_part(image_path))
        except Exception:
            pass

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=contents
    )

    doctor_intro = (
        "Hello, I am your virtual doctor. "
        "Based on what you have shared, here is some general information."
    )

    doctor_text = f"{doctor_intro} {response.text}"
    doctor_audio = generate_doctor_voice(doctor_text)

    return patient_text, doctor_text, doctor_audio
