# from huggingface_hub import InferenceClient
#
# client = InferenceClient(model="Sergiu2404/fin_tinybert")
# response = client.text_classification("Company profits are rising.")
# print(response)





from transformers import pipeline

# Initialize the sentiment analysis pipeline with your model
sentiment_analyzer = pipeline(
    "text-classification",  # or "sentiment-analysis" depending on how your model was uploaded
    model="Sergiu2404/fin_tinybert"
)

# Example texts
texts = [
    "The company's earnings exceeded expectations.",
    "They faced major losses this quarter.",
    "Stock prices remained the same.",
    "AMD reduced debt significantly, improves balance sheet"
]

# Process all examples
for text in texts:
    result = sentiment_analyzer(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    print("-" * 50)