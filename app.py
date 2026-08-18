import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ==========================================================
# 1. INITIALIZE TRANSFORMER MODEL (CACHED FOR SPEED)
# ==========================================================
MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"

print("[*] Downloading and initializing Transformer Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

class_names = ['Reliable', 'Fake']
explainer = LimeTextExplainer(class_names=class_names)

# ==========================================================
# 2. MODEL PREDICTOR FOR LIME
# ==========================================================
def transformer_predictor(texts):
    probabilities = []
    for text in texts:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()
            # Handle edge case for single-word evaluation arrays
            if probs.ndim == 0:
                probs = np.array([probs, 1 - probs]) if predicted_class_idx == 0 else np.array([1 - probs, probs])
            probabilities.append(probs)
    return np.array(probabilities)

# ==========================================================
# 3. CORE PROCESSING FUNCTION FOR THE WEB APP
# ==========================================================
def analyze_article(text):
    if not text.strip():
        return "Please enter text to analyze.", "No data."

    # Get predictions
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().numpy()

    predicted_class_idx = np.argmax(probs)
    label = class_names[predicted_class_idx].upper()
    confidence = probs[predicted_class_idx] * 100

    # Generate XAI Explanations
    explanation = explainer.explain_instance(text, transformer_predictor, num_features=5)

    # Format the text output breakdown
    metrics_summary = f"⚖️ DETECTED LABEL: {label}\n🎯 CONFIDENCE SCORE: {confidence:.2f}%"

    explanation_output = "🔍 EXPLAINABLE AI (XAI) WORD WEIGHTS:\n"
    explanation_output += "Words that triggered the system's decision:\n\n"
    for word, weight in explanation.as_list():
        direction = "🔴 [Points to FAKE]" if weight > 0 else "🟢 [Points to RELIABLE]"
        explanation_output += f"• '{word}' → Weight: {weight:+.4f} {direction}\n"

    return metrics_summary, explanation_output

# ==========================================================
# 4. GRADIO WEB INTERFACE SETUP
# ==========================================================
interface = gr.Interface(
    fn=analyze_article,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste a news article headline or paragraph here to analyze...",
        label="Input News Article"
    ),
    outputs=[
        gr.Textbox(label="Analysis Metrics"),
        gr.Textbox(label="Explainable AI (XAI) Breakdown")
    ],
    title="IntelliGen Anti-Misinformation Engine",
    description="Enterprise-grade Fake News Detection tool combining deep Transformer contexts with Explainable AI transparency.",
    theme="soft"
)

# Launching inside Google Colab (inline and via a public shareable link)
interface.launch(inline=True, share=True)
