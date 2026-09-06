# Transformer Sentiment Analysis

## Overview

An AI-powered sentiment analysis application that uses a Transformer-based DistilBERT model to classify text as positive or negative.

The application supports both individual text classification and batch sentiment analysis through CSV file uploads, along with model evaluation metrics and visualizations.

## Features

- Single-text sentiment classification
- Batch sentiment analysis using CSV files
- DistilBERT Transformer model
- Confidence score for predictions
- Accuracy, Precision, Recall, and F1 Score
- Confusion Matrix visualization
- Sentiment distribution visualization
- Downloadable sentiment analysis report
- Streamlit web interface

## Technologies

- Python
- Hugging Face Transformers
- DistilBERT
- PyTorch
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Git & GitHub

## Project Structure

```text
transformer-sentiment-analysis/
├── data/
├── models/
├── src/
│   └── batch_sentiment.py
├── pages/
│   ├── 1_Text_Classification.py
│   └── 2_File_Analysis.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/namirag/transformer-sentiment-analysis.git
cd transformer-sentiment-analysis

Create the virtual environment
python -m venv tsavenv

Activate the virtual environment
Windows:
tsavenv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run the application
python -m streamlit run app.py

```

## Model Evaluation

The model can be evaluated using labeled CSV data.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The application also provides a sentiment distribution chart and allows users to download the analyzed results as a CSV report.

## Author

Namira Ganam
