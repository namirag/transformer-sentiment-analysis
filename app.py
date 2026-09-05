import streamlit as st
from transformers import pipeline
from src.batch_sentiment import analyze_csv

#loading the pipeline
#caching the model to avoid reloading it every time the app runs
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

#ui streamlit interface
st.title("Transformer Sentiment Analysis")
st.info(
    "Model: DistilBERT | Framework: Hugging Face Transformers | "
    "Task: Text Sentiment Classification"
)


#adding the file.
st.subheader("Batch Sentiment Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#setting dataframe to display the results of the analysis
if uploaded_file:
    result_df, accuracy, precision, recall, f1 = analyze_csv(uploaded_file)

    st.write("Analysis Results")
    st.dataframe(result_df)
    
    #model evaluation metrics
    st.write("### Model Evaluation")

    st.write("Accuracy:", f"{accuracy:.2%}")
    st.write("Precision:", f"{precision:.2%}")
    st.write("Recall:", f"{recall:.2%}")
    st.write("F1 Score:", f"{f1:.2%}")


    # adding a summary of the sentiment analysis
    st.write("*Sentiment Analysis Summary.*")
    
    total_reviews = len(result_df)
    positive_reviews = (result_df["sentiment"] == "POSITIVE").sum()
    negative_reviews = (result_df["sentiment"] == "NEGATIVE").sum()
    info_reviews = (result_df["sentiment"] == "NEUTRAL").sum()
    
    st.write("Total Reviews:", total_reviews)
    st.write("Positive Reviews:", positive_reviews)
    st.write("Negative Reviews:", negative_reviews)
    st.write("Neutral Reviews:", info_reviews)
    
    # adding bar chart to visualize the sentiment distribution.
    st.write('###Sentiment Distribution')
    sentiment_counts = result_df["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)
    
    csv = result_df.to_csv(index=False)
    
    st.download_button(
        label = "Download results",
        data = csv,
        file_name = "sentiment_report.csv",
        mime = "text/csv"
    )

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
        