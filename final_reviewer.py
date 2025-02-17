import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load Model and Tokenizer
model_path = "C:/Users/manan/OneDrive/Desktop/saved_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

def predict_review(review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"

# Streamlit UI
st.title("🛍️ Fake Product Review Detector")
st.write("Enter a product review below and check if it's real or fake.")

review_input = st.text_area("Enter your review:", "")
if st.button("Predict Review"):
    if review_input.strip():
        result = predict_review(review_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter a review before predicting.")
