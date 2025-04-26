from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

model_name = "yiyanghkust/finbert-tone"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    # [negative, neutral, positive]
    # score = probs[2] - probs[0]
    return (probs[0], probs[1], probs[2])

text = "The company's profits exceeded expectations, leading to a surge in stock prices."
score = get_sentiment_score(text)
print(f"Scores: {score[0]:.6f}, {score[1]:.6f}, {score[2]:.6f}")
