import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.batch_sentiment import analyze_csv

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
    
    # confusion matrix
    cm = confusion_matrix(
    result_df["label"],
    result_df["sentiment"])
    
    # st.write("### Confusion Matrix")
    # st.dataframe(
    #     pd.DataFrame(
    #         cm,
    #         index=["Actual NEGATIVE", "Actual POSITIVE"],
    #         columns=["Predicted NEGATIVE", "Predicted POSITIVE"]
    #     )
    # )
    
    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted NEGATIVE", "Predicted POSITIVE"])
    ax.set_yticklabels(["Actual NEGATIVE", "Actual POSITIVE"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)


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
