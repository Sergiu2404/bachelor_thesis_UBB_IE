import torch
from transformers import AutoTokenizer
from fin_tinybert_pytorch import TinyFinBERTRegressor


class InferenceAPI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyFinBERTRegressor()
        self.model.load_state_dict(torch.load("./saved_model/pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("./saved_model")

    def __call__(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        results = []
        for text in inputs:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True,
                                     padding='max_length', max_length=128)
            encoded = {k: v.to(self.device) for k, v in encoded.items() if k != "token_type_ids"}

            with torch.no_grad():
                score = self.model(**encoded)["score"].item()

            sentiment = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"

            results.append({
                "label": sentiment,
                "score": round(score, 4)
            })

        if len(results) == 1:
            return results[0]
        return results