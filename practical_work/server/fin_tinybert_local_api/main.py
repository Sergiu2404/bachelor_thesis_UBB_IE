from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import re
import spacy
from nltk.corpus import stopwords
import os
from model import TinyFinBERTRegressor

app = FastAPI()

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

MODEL_DIR = "E:/saved_models/sentiment_analysis_fine_tuned_tinybert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = TinyFinBERTRegressor()
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "regressor_model.pt"), map_location=DEVICE), strict=False)
model.to(DEVICE)
model.eval()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_.strip()
    ]
    return ' '.join(tokens)


class InputText(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Sentiment Regressor Local API"}
@app.post("/predict")
async def predict_sentiment(input_text: InputText):
    try:
        clean_text = preprocess_text(input_text.text)
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            score = model(**inputs)["score"].item()

        print(f"sentiment score {score}")

        return {
            "original_text": input_text.text,
            "processed_text": clean_text,
            "sentiment_score": round(score, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))