import os
import tempfile
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from google import genai
from google.genai import types


# ---------- GEMINI SETUP ----------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing")

client = genai.Client(api_key=API_KEY)


# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = (
    "You are acting as a professional virtual doctor for educational purposes only. "
    "Carefully observe the user's description and any provided image. "
    "Explain what you notice in simple, clear language. "
    "Do not provide a medical diagnosis, prescriptions, or definitive treatment plans. "
    "You may suggest general care or lifestyle-related measures if appropriate. "
    "Respond in a single concise paragraph. "
    "Do not say that you are an AI model."
)


# ---------- GEMINI PART HELPERS ----------
def text_part(text: str):
    return types.Part.from_text(text=text)


def image_part(image_path: str):
    with open(image_path, "rb") as f:
        return types.Part.from_bytes(
            data=f.read(),
            mime_type="image/jpeg"
        )


# ---------- VOICE TO TEXT ----------
def voice_to_text(mic_audio):
    """
    Converts microphone audio from Gradio (numpy format)
    into text using Google Speech Recognition.
    """
    if mic_audio is None:
        return None

    sample_rate, audio_np = mic_audio
    audio_np = audio_np.astype(np.int16)

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav.write(temp_wav.name, sample_rate, audio_np)
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)

    os.remove(temp_wav.name)

    try:
        return recognizer.recognize_google(audio_data)
    except Exception:
        return None


# ---------- MAIN BRAIN PIPELINE ----------
def brain_pipeline(text_input=None, mic_audio=None, image_path=None):
    """
    Main reasoning pipeline for the AI Doctor.
    Accepts text, voice, and optional image input.
    Returns a Gemini-generated medical observation.
    """

    parts = [text_part(SYSTEM_PROMPT)]

    # 🎙️ Voice has priority over text
    if mic_audio is not None:
        spoken_text = voice_to_text(mic_audio)
        if spoken_text:
            parts.append(text_part(spoken_text))
    elif text_input and text_input.strip():
        parts.append(text_part(text_input))
    else:
        return "Please provide a description using text or voice."

    # 📷 Optional image input
    if image_path:
        try:
            parts.append(image_part(image_path))
        except Exception:
            pass

    contents = [
        types.Content(
            role="user",
            parts=parts
        )
    ]

    try:
        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=contents
        )
        return response.text

    except Exception as e:
        print("Gemini error:", e)
        return "I encountered an issue while processing the request. Please try again."
