import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define response generation function
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# Define Gradio interface functions
def disease_prediction(symptoms):
    prompt = f"Given the following symptoms, provide possible medical conditions and general medication suggestions. Always emphasize the importance of consulting a healthcare professional:\n{symptoms}"
    return generate_response(prompt, max_length=1200)

def treatment_plan(condition, age, gender, history):
    prompt = (
        f"Generate personalized treatment suggestions for the following patient information. "
        f"Include home remedies and general medication guidelines.\n"
        f"Condition: {condition}\nAge: {age}\nGender: {gender}\nMedical History: {history}"
    )
    return generate_response(prompt, max_length=1300)

# Build Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## 🏥 Medical AI Assistant")
    gr.Markdown("**Disclaimer:** This is for informational purposes only. Always consult healthcare professionals for medical advice.")

    with gr.Tabs():
        with gr.TabItem("Disease Prediction"):
            with gr.Row():
                with gr.Column():
                    symptom_input = gr.Textbox(
                        placeholder="e.g., fever, headache, cough, fatigue...",
                        label="Enter Symptoms",
                        lines=4
                    )
                    predict_btn = gr.Button("Analyze Symptoms")
                with gr.Column():
                    prediction_output = gr.Textbox(
                        label="Possible Conditions & Recommendations",
                        lines=20
                    )
            predict_btn.click(fn=disease_prediction, inputs=symptom_input, outputs=prediction_output)

        with gr.TabItem("Treatment Plan"):
            with gr.Row():
                with gr.Column():
                    condition_input = gr.Textbox(label="Medical Condition", placeholder="e.g., diabetes, hypertension...", lines=2)
                    age_input = gr.Number(label="Age", value=30)
                    gender_input = gr.Radio(choices=["Male", "Female", "Other"], label="Gender")
                    history_input = gr.Textbox(label="Medical History", placeholder="Previous conditions, allergies, medications...", lines=4)
                    plan_btn = gr.Button("Generate Treatment Plan")
                with gr.Column():
                    plan_output = gr.Textbox(label="Personalized Treatment Plan", lines=20)
            plan_btn.click(fn=treatment_plan, inputs=[condition_input, age_input, gender_input, history_input], outputs=plan_output)

# Launch app
app.launch(share=True)