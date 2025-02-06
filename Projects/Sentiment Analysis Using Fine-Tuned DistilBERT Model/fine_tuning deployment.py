import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load fine-tuned model and tokenizer
model_path = "C:\\Users\\Admin\\Downloads\\Fine Tuned Model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set page config for better UX
st.set_page_config(page_title="Sentiment Analysis Classifier", page_icon="📝", layout="centered")

# Customizing the layout with markdown and a bit of CSS
st.markdown("""
    <style>
    /* Set the background color of the whole page */
    body {
        background-color: #f0f8ff;  /* Light blue background */
    }
    
    .title {
        text-align: center;
        font-size: 36px;
        color: #1f77b4;
        font-weight: bold;
    }
    .subheading {
        text-align: center;
        font-size: 18px;
        color: #333;
    }
    .main-container {
        background-color: #ffffff;  /* White background for content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .input-box {
        font-size: 18px;
        border-radius: 10px;
        padding: 15px;
        width: 100%;
    }
    .btn {
        background-color: #1f77b4;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 18px;
        width: 100%;
    }
    .btn:hover {
        background-color: #174c72;
    }
    .model-info {
        font-size: 16px;
        color: #555;
        text-align: center;
        margin-top: 20px;
    }
    .model-headline {
        font-size: 22px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="title">Sentiment Analysis Classifier 📝</div>', unsafe_allow_html=True)
st.markdown('<div class="subheading">Classify the sentiment of your text as Positive or Negative! 😄</div>', unsafe_allow_html=True)

# Input text area for user review
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    user_input = st.text_area("Enter a review:", height=150, max_chars=1000, key="review", help="Enter a movie review, product review, or any other text to analyze its sentiment.")
    st.markdown('</div>', unsafe_allow_html=True)

# Button to trigger classification
if st.button("Classify Sentiment", key="classify", help="Click to classify the sentiment of the entered text"):
    if user_input:
        with st.spinner("Classifying... please wait!"):
            # Tokenizing the input text
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax().item()

            # Displaying results with emojis
            if predicted_class == 1:
                st.success("Positive Sentiment 😊", icon="✅")
            else:
                st.error("Negative Sentiment 😞", icon="❌")
    else:
        st.warning("Please enter some text to classify.", icon="⚠️")

# Optional: Clear button to reset input
if st.button("Clear Input", key="clear"):
    st.session_state["review"] = ""  # Clear the input field
    st.experimental_rerun()

# Model Info Section
st.markdown("<div class='model-headline'>Model Details:</div>", unsafe_allow_html=True)
st.markdown("""
    <div class='model-info'>
        This sentiment analysis classifier is based on the model "<b>distilbert-base-uncased</b>", which has been fine-tuned specifically for sentiment classification tasks. The model was fine-tuned on a "<b>IMDb</b>" dataset to classify sentiments into positive and negative classes. 
    </div>
""", unsafe_allow_html=True)
