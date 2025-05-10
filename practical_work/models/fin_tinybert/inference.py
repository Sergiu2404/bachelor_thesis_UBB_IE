import torch
from transformers import AutoTokenizer
from fin_tinybert_pytorch import TinyFinBERTRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyFinBERTRegressor()
model.load_state_dict(torch.load("./saved_model/pytorch_model.bin", map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./saved_model")


def pipeline(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        score = model(**inputs)["score"].item()

    sentiment = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"

    return [{
        "label": sentiment,
        "score": round(score, 4)
    }]


if __name__ == "__main__":
    texts = [
        "The stock price soared after the earnings report.",
        "The company reported significant losses this quarter.",
        "There was no noticeable change in performance."
    ]

    predictions = pipeline("The stock price soared after the earnings report.")[0]
    print(f"sentiment: {predictions['label']}, score: {predictions['score']}")