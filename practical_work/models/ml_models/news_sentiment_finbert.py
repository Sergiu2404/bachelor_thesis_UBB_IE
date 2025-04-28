# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# import torch.nn.functional as F
#
# tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
# model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
#
# text_pos = "The company's quarterly earnings exceeded expectations, leading to a surge in stock prices." # 0.9
# text_neg = "The company just reported great losses and this may lead to a steep fall in stock prices." # -0.9
# text_neu = "The company's stock prices kept the same trend for the last months." # 0.2
#
# inputs = tokenizer(text_neu, return_tensors="pt", truncation=True, max_length=512)
#
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probabilities = F.softmax(logits, dim=1)
#
# sentiment_scores = torch.tensor([1, -1, 0])
# weighted_score = torch.sum(probabilities * sentiment_scores)
# print(f"Sentiment score: {weighted_score.item():.3f}")




from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

text_pos2 = "The company's quarterly earnings exceeded expectations, leading to a surge in stock prices."
text_neg2 = "The company just reported great losses and this may lead to a steep fall in stock prices."
text_neu2 = "The company's stock prices kept the same trend for the last months."
text = text_pos2

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)

sentiment_scores = torch.tensor([-1, 1])
weighted_score = torch.sum(probabilities * sentiment_scores)
print(f"Sentiment score: {weighted_score.item():.3f}")
