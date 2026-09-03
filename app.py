import streamlit as st
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

st.title("Transformer Sentiment Analysis")
st.info(
    "Model: DistilBERT | Framework: Hugging Face Transformers | "
    "Task: Text Sentiment Classification"
)
st.write("Enter text below to analyze its sentiment.")

text = st.text_area("Enter your text:", height=150)

if st.button("Analyze Sentiment"):
    if text:
        result = sentiment_pipeline(text)[0]

        st.subheader("Result")
        sentiment_result = result["label"]
        
        if sentiment_result == "POSITIVE":
            st.success("Sentiment: Positive")
        elif sentiment_result == "NEGATIVE":
            st.error("Sentiment: Negative")
        else:
            st.info("Sentiment: Neutral")
        
        confidence_score = result["score"]
        st.write("**Confidence:**", f"{confidence_score * 100:.2f}%")
    else:
        st.warning("Please enter some text.")