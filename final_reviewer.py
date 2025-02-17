from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
model_path = ("C:/Users/manan/OneDrive/Desktop/saved_model")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)


def predict_review(review):
    # Tokenize the input review
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    # Get the predicted label (0 for Fake, 1 for Real)
    prediction = torch.argmax(logits, dim=1).item()
    return "Real" if prediction == 1 else "Fake"



sample_review = "This is an extremely outstanding product"
print(f"Review: {sample_review}\nPrediction: {predict_review(sample_review)}")