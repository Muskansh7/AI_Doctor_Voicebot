# 🩺 AI Doctor Voicebot

An AI-powered healthcare assistant that combines **speech recognition**, **medical image understanding**, and **Large Language Models (LLMs)** to provide intelligent and interactive healthcare support.

The application allows users to speak their symptoms, upload medical images, and receive contextual medical insights in both text and voice format through an intuitive interface.

> **Disclaimer:** This project is intended for educational and research purposes only. It does not replace professional medical advice, diagnosis, or treatment.

---

## ✨ Features

* 🎤 **Voice Input**

  * Ask medical questions naturally using speech.

* 🖼️ **Medical Image Analysis**

  * Upload images and receive AI-generated insights.

* 🧠 **LLM-Powered Medical Reasoning**

  * Generates contextual responses using modern Large Language Models.

* 🔊 **Voice Output**

  * Converts AI responses into natural-sounding speech.

* 🌐 **Interactive Interface**

  * User-friendly web interface for seamless interactions.

* ⚡ **Real-Time Processing**

  * Fast and responsive AI inference pipeline.

---

## 🏗️ System Architecture

```text
User Voice / Medical Image
            │
            ▼
 ┌─────────────────────┐
 │ Input Processing    │
 │ - Speech to Text    │
 │ - Image Processing  │
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ AI Doctor Engine    │
 │ - LLM Reasoning     │
 │ - Context Analysis  │
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │ Response Generation │
 │ - Text Response     │
 │ - Text to Speech    │
 └─────────────────────┘
            │
            ▼
         User
```

---

## 🛠️ Tech Stack

### Languages

* Python

### AI / ML

* Large Language Models (LLMs)
* Natural Language Processing (NLP)
* Speech Recognition
* Computer Vision
* Generative AI

### Frameworks & Libraries

* Gradio / Streamlit
* Hugging Face
* Groq API
* OpenAI Whisper
* gTTS / ElevenLabs

### Tools

* Git
* GitHub
* Python Virtual Environment

---

## 📂 Project Structure

```text
AI_Doctor_Voicebot/
│
├── app.py                 # Main application
├── brain.py               # AI reasoning engine
├── patient_query.py       # Voice input handling
├── doctor_response.py     # Response generation
├── requirements.txt       # Dependencies
├── assets/                # Images and media
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Muskansh7/AI_Doctor_Voicebot.git

cd AI_Doctor_Voicebot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

### 5. Run the Application

```bash
python app.py
```

or

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

* Preliminary healthcare assistance
* Voice-based medical interactions
* Medical image understanding
* Healthcare accessibility solutions
* AI and Generative AI research

---

## 📈 Future Improvements

* Multi-language support
* Patient history and memory
* RAG-based medical knowledge retrieval
* Appointment scheduling
* Cloud deployment and monitoring

---

## 👩‍💻 Author

**Muskan Sharma**

* GitHub: https://github.com/Muskansh7
* LinkedIn: https://www.linkedin.com/in/muskan-sharma-91025b297/

If you found this project useful, consider giving it a ⭐ on GitHub.

## website link

https://huggingface.co/spaces/muskansh7/ai-doctor-voicebot
