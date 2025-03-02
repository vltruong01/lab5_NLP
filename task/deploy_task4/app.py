import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load your trained model and tokenizer
model_name = "vltruong01/dpo_model_lr5e-05_bs16_epoch1"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prediction function
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    # Map the numeric prediction to a human-readable label
    label_map = {
        0: "Negative Sentiment",   # Example label for 0
        1: "Positive Sentiment",   # Example label for 1
    }
    
    return label_map.get(prediction, "Unknown Category")  # Default to "Unknown Category"

# Set up the Streamlit web interface
st.title("Model Demo: Text Classification")
st.write("This app allows you to input text and see the model's response.")

# Get input from the user
input_text = st.text_area("Enter Text Here:")

# Display the prediction when the user submits the input
if st.button("Get Prediction"):
    if input_text:
        prediction = predict(input_text)
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter some text to get a prediction.")