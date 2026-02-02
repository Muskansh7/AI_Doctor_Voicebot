import os
from datetime import datetime
from gtts import gTTS
from google import genai


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


# ---------- STORAGE PATHS ----------
BASE_RECORDINGS_DIR = "recordings"
DOCTOR_AUDIO_DIR = os.path.join(BASE_RECORDINGS_DIR, "doctor_responses")

os.makedirs(DOCTOR_AUDIO_DIR, exist_ok=True)


# ---------- GEMINI RESPONSE (BRAIN OF DOCTOR) ----------
def get_gemini_response(user_text: str) -> str:
    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=[SYSTEM_PROMPT, user_text]
    )
    return response.text


# ---------- TEXT TO SPEECH (VOICE OF DOCTOR) ----------
def generate_doctor_voice(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = os.path.join(
        DOCTOR_AUDIO_DIR,
        f"doctor_response_{timestamp}.mp3"
    )

    tts = gTTS(
        text=text,
        lang="en",
        slow=False
    )
    tts.save(audio_path)

    return audio_path


# ---------- MAIN PIPELINE FOR GRADIO ----------
def doctor_pipeline(patient_text: str):
    if not patient_text.strip():
        return "Please describe your symptoms.", None

    doctor_intro = (
        "Hello, I am your virtual doctor. "
        "Based on what you have shared, here is some general information."
    )

    doctor_response = get_gemini_response(patient_text)
    full_doctor_text = f"{doctor_intro} {doctor_response}"

    audio_path = generate_doctor_voice(full_doctor_text)

    return full_doctor_text, audio_path
