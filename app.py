import os
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ==============================================================================
# 1. CORE INTELLIGEN ENGINE DEFINITION
# ==============================================================================
class IntelliGenEngine:
    def __init__(self, model_name: str = "ErfanMoosaviMonazzah/bert-tiny-fake-news-detection"):
        """
        Initializes the model core and vocabulary tokenizers.
        """
        # Auto-detect target runtime environment compute hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pull required model layers down from Hugging Face cache layers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Freeze layers into inference evaluation mode
        
        # Dynamically pull mapped labels directly out of network configuration metrics
        self.id2label = {int(k): v.upper() for k, v in self.model.config.id2label.items()}
        self.class_names = [self.id2label[i] for i in sorted(self.id2label.keys())]
        
        # Load localized transparency matrices
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_probabilities(self, texts: list) -> np.ndarray:
        """
        Transforms raw user-pasted text strings into continuous math tensors.
        """
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Map structural raw logit scores to probability distribution matrices
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return probabilities.cpu().numpy()

    def analyze_text(self, text: str):
        """
        Processes a block of content, extracts verdicts, and structures XAI maps.
        """
        if not text.strip():
            return None, None, []
            
        probs = self.predict_probabilities([text])
        pred_idx = np.argmax(probs)
        verdict = self.class_names[pred_idx]
        confidence = probs[pred_idx] * 100
        
        # Locally sample data neighborhoods using the LIME engine bounds
        exp = self.explainer.explain_instance(
            text, 
            self.predict_probabilities, 
            num_features=8, 
            num_samples=100
        )
        return verdict, confidence, exp.as_list()

# ==============================================================================
# 2. STREAMLIT FRONT-END DASHBOARD ROUTINES
# ==============================================================================
st.set_page_config(page_title="IntelliGen Anti-Misinformation Engine", page_icon="🛡️", layout="wide")

st.title("🛡️ IntelliGen Anti-Misinformation Engine")
st.caption("Enterprise text verification screen powered by an optimized bert-tiny backbone and localized XAI transparency maps.")

# Cache the AI engine in memory so it doesn't reload on every UI click
if "engine" not in st.session_state:
    with st.spinner("Downloading and caching bert-tiny model layers from Hugging Face..."):
        st.session_state.engine = IntelliGenEngine()

# Standard dashboard text entry layout region
input_text = st.text_area(
    label="Article Text Input Payload", 
    height=220, 
    placeholder="Paste news content or unstructured article scripts here for classification screening..."
)

if st.button("Execute Verification Screen", type="primary"):
    if input_text:
        with st.spinner("Processing text tokens and mapping linear decision bounds..."):
            verdict, confidence, explanations = st.session_state.engine.analyze_text(input_text)
            
        if verdict:
            # Render layout metrics side by side
            col1, col2 = st.columns(2)
            with col1:
                st.metric("System Verdict Classification Result", verdict)
            with col2:
                st.metric("Statistical Predictor Confidence", f"{confidence:.2f}%")
            
            st.subheader("XAI Multi-Weight Transparency Blueprint")
            st.write("Linguistic feature adjustments pushing or pulling target decisions:")
            
            # Map word attribution directions cleanly
            for word, weight in explanations:
                if "FAKE" in verdict and weight > 0:
                    st.error(f"🔴 **'{word}'** -> Drives toward FAKE classification (Impact Score: {abs(weight):.4f})")
                else:
                    st.success(f"🟢 **'{word}'** -> Anchors toward RELIABLE status (Impact Score: {abs(weight):.4f})")
    else:
        st.warning("Please enter a valid text block to run analysis metrics.")
