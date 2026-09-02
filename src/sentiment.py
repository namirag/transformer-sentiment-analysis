from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
text =  input("Enter a sentence:")
results = sentiment_pipeline(text)
print("Sentiment:", results[0]["label"])
print("Confidence:", f"{results[0]['score']*100:.2f}%")
