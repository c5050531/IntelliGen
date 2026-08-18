# IntelliGen
Understand the fundamental concepts of artificial intelligence and machine learning.  • Apply and evaluate machine learning to answer complex questions within a range of contexts that are relevant to industry and business.   
#  IntelliGen Anti-Misinformation Engine

An enterprise-grade Fake News Detection tool combining deep Transformer architecture with Explainable AI (XAI) transparency. This application uses a fine-tuned tiny BERT model to classify news texts and leverages LIME to show exactly which words influenced the decision.

##  Features
* **Transformer Core:** Powered by a fine-tuned `bert-tiny` sequence classification model.
* **Explainable AI:** Uses LIME text explainers to calculate weight values for high-impact words.
* **Web UI:** Accessible, lightweight browser interface built on Gradio.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd intelligen-anti-misinformation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

##  How It Works
* ** Red Weights:** Words contributing heavily toward a **FAKE** classification.
* ** Green Weights:** Words anchoring the context toward a **RELIABLE** status.
