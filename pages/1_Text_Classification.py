import streamlit as st
from transformers import pipeline

#loading the pipeline
#caching the model to avoid reloading it every time the app runs
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

#text analysis-section
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