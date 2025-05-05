# import requests
#
# API_URL = "https://api-inference.huggingface.co/models/sergiu2404/fin_tinybert_space"
# headers = {"Authorization": f"Bearer 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"}
#
# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()
#
# # Try with a sample text
# output = query({
#     "inputs": "The company just announced profit exceeds expectation for this year."
# })
#
# print(output)


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Assuming it's a sequence classification model for sentiment
model_name = "sergiu2404/fin_tinybert"  # Adjust if the model name is different
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForSequenceClassification.from_pretrained(model_name, token="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")

text = "The company just announced profit exceeds expectation for this year."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# Map to labels if you know them
# labels = ["negative", "neutral", "positive"]
# predicted_class = predictions.argmax().item()
# print(f"Predicted class: {labels[predicted_class]} with score: {predictions[0][predicted_class].item():.4f}")