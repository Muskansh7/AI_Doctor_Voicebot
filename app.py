import gradio as gr
from brain_of_the_doctor import brain_pipeline
from voice_of_the_doctor import generate_doctor_voice


def ai_doctor_pipeline(patient_text, mic_audio, image_path):
    try:
        # 🧠 Doctor reasoning
        doctor_reasoning = brain_pipeline(
            text_input=patient_text,
            mic_audio=mic_audio,
            image_path=image_path
        )

        if not doctor_reasoning:
            return "Please provide symptoms using text, voice, or image.", None

        # 🩺 Doctor intro
        doctor_intro = (
            "Hello, I am your virtual doctor. "
            "Based on what you have shared, here is some general information."
        )

        full_doctor_text = f"{doctor_intro} {doctor_reasoning}"

        # 🗣️ Doctor voice
        audio_path = generate_doctor_voice(full_doctor_text)

        return full_doctor_text, audio_path

    except Exception as e:
        print("Pipeline error:", e)
        return "Something went wrong. Please try again.", None


with gr.Blocks(title="AI Doctor Voice Assistant") as demo:
    gr.Markdown("## 🩺 AI Doctor Voice Assistant")
    gr.Markdown(
        "A smart virtual medical assistant for educational purposes"
    )

    # -------- INPUTS --------
    patient_text = gr.Textbox(
        label="🧑 Patient Description",
        placeholder="Describe your symptoms...",
        lines=4
    )

    mic_audio = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="🎙️ Patient Voice Input"
    )

    image_input = gr.Image(
        type="filepath",
        label="📷 Upload Image (Optional)",
        height=200
    )

    # -------- OUTPUTS --------
    doctor_text = gr.Textbox(
        label="🧠 Doctor Response",
        lines=6
    )

    doctor_audio = gr.Audio(
        label="🗣️ Doctor Voice",
        type="filepath"
    )

    consult_btn = gr.Button("Consult Doctor")

    consult_btn.click(
        fn=ai_doctor_pipeline,
        inputs=[patient_text, mic_audio, image_input],
        outputs=[doctor_text, doctor_audio],
        show_progress=True   # ✅ loading spinner
    )


# ✅ IMPORTANT: disable SSR (fixes asyncio error)
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
