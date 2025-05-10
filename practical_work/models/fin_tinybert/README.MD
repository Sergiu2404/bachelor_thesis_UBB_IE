---
language: en
license: mit
tags:
  - text-classification
  - finance
  - sentiment-analysis
datasets:
  - financial_sentiment
metrics:
  - accuracy
pipeline_tag: text-classification
---

# FinTinyBERT - Financial Sentiment Analysis Model

This model performs sentiment analysis on financial texts, classifying them as positive, negative, or neutral.

## Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="Sergiu2404/fin_tinybert")
result = classifier("Company profits are rising.")
print(result)
