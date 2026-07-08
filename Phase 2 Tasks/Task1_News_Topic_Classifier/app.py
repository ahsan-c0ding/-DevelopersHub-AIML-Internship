"""
Streamlit app for interactive news topic classification using a BERT model
fine-tuned on AG News.

Expects a fine-tuned checkpoint at MODEL_DIR, produced by running
News_Topic_Classifier_BERT.ipynb (which saves to ./bert-ag-news/final).

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

MODEL_DIR = "./bert-ag-news/final"  # path to the fine-tuned model from the training notebook
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAX_LENGTH = 64

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")


@st.cache_resource(show_spinner="Loading fine-tuned BERT model...")
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    return dict(zip(LABEL_NAMES, probs))


st.title("📰 News Topic Classifier")
st.caption(
    "Fine-tuned BERT (bert-base-uncased) on the AG News dataset — "
    "classifies headlines into World, Sports, Business, or Sci/Tech."
)

try:
    tokenizer, model = load_model()
except OSError:
    st.error(
        f"No fine-tuned model found at `{MODEL_DIR}`.\n\n"
        "Run `News_Topic_Classifier_BERT.ipynb` first (or point MODEL_DIR at your "
        "saved checkpoint) — the notebook saves the fine-tuned model there after training."
    )
    st.stop()

text = st.text_area(
    "Enter a news headline or short description",
    placeholder="e.g. Apple unveils new AI chip for next-generation iPhones",
    height=100,
)

col1, col2 = st.columns([1, 4])
classify_clicked = col1.button("Classify", type="primary")

if classify_clicked and text.strip():
    with st.spinner("Classifying..."):
        probs = predict(text, tokenizer, model)

    predicted = max(probs, key=probs.get)
    st.success(f"**Predicted category: {predicted}**  ({probs[predicted]:.1%} confidence)")

    df = pd.DataFrame({"Category": list(probs.keys()), "Confidence": list(probs.values())})
    df = df.sort_values("Confidence", ascending=False).set_index("Category")
    st.bar_chart(df)

    with st.expander("Raw probabilities"):
        st.json({k: round(v, 4) for k, v in probs.items()})

elif classify_clicked:
    st.warning("Please enter some text first.")

st.divider()

with st.expander("Try example headlines"):
    examples = [
        "Federal Reserve raises interest rates amid inflation concerns",
        "Manchester United secures dramatic last-minute victory",
        "NASA telescope captures stunning new images of distant galaxy",
        "United Nations calls for ceasefire amid escalating conflict",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

st.caption("Model: bert-base-uncased fine-tuned on AG News (4 classes) · AI/ML Internship, DevelopersHub Corporation")
