import torch
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_csv(file):
    df = pd.read_csv(file)
    
    results = sentiment_pipeline(df["text"].astype(str).tolist())

    df["sentiment"] = [result["label"] for result in results]    
    
    df["confidence"] = [result["score"] for result in results]
    
    accuracy = accuracy_score(df["label"], df["sentiment"])
    precision = precision_score(df["label"], df["sentiment"], average="weighted")
    recall = recall_score(df["label"], df["sentiment"], average="weighted")
    f1 = f1_score(df["label"], df["sentiment"], average="weighted")
    
    
    return df, accuracy, precision, recall, f1

if __name__ == "__main__":
    df, accuracy, precision, recall, f1 = analyze_csv("data/test_reviews.csv")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print(df)
    