import os
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ==============================================================================
# 1. CORE INTELLIGEN CORE ENGINE DEFINITION
# ==============================================================================
class IntelliGenEngine:
    def __init__(self, model_name: str = "ErfanMoosaviMonazzah/bert-tiny-fake-news-detection"):
        """
        Initializes the model core, tokenizers, and explainability frameworks.
        """
        # Auto-detect target runtime environment compute hardware (CPU-safe for Streamlit Cloud)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pull model layers down from Hugging Face model registry
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Freeze weights into evaluation mode for stable inference
        
        # Extract classification labels dynamically from model configuration parameters
        self.id2label = {int(k): v.upper() for k, v in self.model.config.id2label.items()}
        self.class_names = [self.id2label[i] for i in sorted(self.id2label.keys())]
        
        # Initialize Localized Feature-Importance Maps (LIME)
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_probabilities(self, texts: list) -> np.ndarray:
        """
        Transforms raw string tokens into structural mathematical tensor arrays
        and applies Softmax to calculate discrete classifications.
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
            # Map structural raw scores to formal probability distributions
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return probabilities.cpu().numpy()

    def analyze_text(self, text: str):
        """
        Processes a block of text, extracts confidence scores, and builds XAI weights.
        """
        if not text.strip():
            return None, None, []
            
        # Extract 2D matrix shape: [[prob_0, prob_1]]
        probs = self.predict_probabilities([text]) 
        
        # Evaluate highest index score location relative to the batch row
        pred_idx = int(np.argmax(probs[0]))
        verdict = self.class_names[pred_idx]
        
        # CRITICAL FIX: Safe indexing into the batch row to extract the scalar confidence
        confidence = float(probs[0][pred_idx]) * 100
        
        # Locally sample alternative text iterations using LIME boundaries
        exp = self.explainer.explain_instance(
            text, 
            self.predict_probabilities, 
            num_features=8, 
            num_samples=100  # Optimized calculation samples for rapid UI responsiveness
        )
        return verdict, confidence, exp.as_list()

# ==============================================================================
# 2. STREAMLIT FRONT-END DASHBOARD ORCHESTRATION
# ==============================================================================
st.set_page_config(
    page_title="IntelliGen Anti-Misinformation Engine", 
    page_icon="🛡️", 
    layout="wide"
)

st.title("🛡️ IntelliGen Anti-Misinformation Engine")
st.caption("Enterprise-grade text screening framework leveraging a fine-tuned bert-tiny backbone and transparent XAI maps.")

# Prevent application from reloading heavy weights from disk on every interface click
if "engine" not in st.session_state:
    with st.spinner("Downloading and caching fine-tuned model layers from Hugging Face..."):
        try:
            st.session_state.engine = IntelliGenEngine()
            st.success("Core algorithmic pipeline successfully loaded and cached.")
        except Exception as e:
            st.error(f"Failed to load underlying network models: {str(e)}")
            st.stop()

# Text input matrix region
input_text = st.text_area(
    label="Article Text Input Payload", 
    height=220, 
    placeholder="Paste news copy or unstructured article strings here for classification screening..."
)

if st.button("Execute Verification Screen", type="primary"):
    if input_text.strip():
        with st.spinner("Processing continuous text data and mapping decision weights..."):
            try:
                verdict, confidence, explanations = st.session_state.engine.analyze_text(input_text)
                
                if verdict:
                    # Display top-level results side by side using scannable metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("System Verdict Classification", verdict)
                    with col2:
                        st.metric("Statistical Predictor Confidence", f"{confidence:.2f}%")
                    
                    st.subheader("XAI Multi-Weight Transparency Blueprint")
                    st.write("Linguistic feature adjustments pushing or pulling target decisions:")
                    st.write("---")
                    
                    # Graphically represent feature weights using colored alerts
                    for word, weight in explanations:
                        # Determine if the feature supports fake news attributes color-neutrally
                        is_fake_signal = ("FAKE" in verdict and weight > 0) or ("REAL" in verdict and weight < 0)
                        
                        if is_fake_signal:
                            st.error(f"🔴 **'{word}'** ── Drives toward FAKE classification (Impact Magnitude: {abs(weight):.4f})")
                        else:
                            st.success(f"🟢 **'{word}'** ── Anchors toward RELIABLE status (Impact Magnitude: {abs(weight):.4f})")
            except Exception as e:
                st.error(f"An error occurred during network inference execution: {str(e)}")
    else:
        st.warning("Please enter a valid text block to run analytics.")
