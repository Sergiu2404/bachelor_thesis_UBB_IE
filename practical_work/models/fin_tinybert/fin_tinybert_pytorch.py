import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy
import re
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, IntervalStrategy

from nltk.corpus import stopwords
import spacy


class TinyFinBERTRegressor(nn.Module):
    def __init__(self, pretrained_model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()
        if pretrained_model:
            self.config = AutoConfig.from_pretrained(pretrained_model)
            self.bert = AutoModel.from_pretrained(pretrained_model, config=self.config)
        else:
            self.config = AutoConfig()
            self.bert = AutoModel(self.config)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

        # Manually register the position_ids buffer to avoid missing key error
        self.bert.embeddings.register_buffer(
            "position_ids",
            torch.arange(512).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        score = self.regressor(cls_output).squeeze()
        loss = F.mse_loss(score, labels) if labels is not None else None
        return {'loss': loss, 'score': score}


def preprocess_texts(texts):
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    negations = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
                 'hardly', 'scarcely', 'barely', "n't", "without", "unless", "nor"}
    stop_words = set(stopwords.words('english')) - negations

    processed = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if token.lemma_.strip()  # token.lemma_
        ]
        processed.append(' '.join(tokens))
    return processed


def load_phrasebank(path):
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()
    sents, scores = [], []
    for line in lines:
        if '@' in line:
            s, l = line.strip().split('@')
            score = 0.0 if l.lower() == 'neutral' else (-1.0 if l.lower() == 'negative' else 1.0)
            sents.append(s)
            scores.append(score)
    return pd.DataFrame({'text': sents, 'score': scores})


def load_words_phrases(path):
    with open(path, 'r', encoding='latin1') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        match = re.search(r',(-?\d+\.?\d*)$', line)
        if match:
            text = line[:match.start()].strip()
            score = float(match.group(1))
            data.append((text, score))
    return pd.DataFrame(data, columns=["text", "score"])


def train_model(phrase_path, words_path, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    phrase_df = load_phrasebank(phrase_path)
    words_df = load_words_phrases(words_path)

    phrase_df['text'] = preprocess_texts(phrase_df['text'])
    words_df['text'] = preprocess_texts(words_df['text'])

    train_phrase, test_phrase = train_test_split(phrase_df, test_size=0.2, random_state=42)
    train_df = pd.concat([train_phrase, words_df])
    test_df = test_phrase.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding='max_length', truncation=True, max_length=128)
        tokens["labels"] = batch["score"]
        return tokens

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir=os.path.join(save_path, "results"),
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    model = TinyFinBERTRegressor().to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: {
            "mse": mean_squared_error(pred.label_ids, pred.predictions),
            "r2": r2_score(pred.label_ids, pred.predictions)
        }
    )

    trainer.train()

    # Save the model and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Handle distributed/parallel training
    torch.save(model_to_save.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize


def evaluate_model(phrase_path, model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    phrase_df = load_phrasebank(phrase_path)
    _, test_df = train_test_split(phrase_df, test_size=0.2, random_state=42)
    test_df['text'] = preprocess_texts(test_df['text'])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TinyFinBERTRegressor()
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    for _, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            score = model(**inputs)["score"].item()
        y_scores.append(score)
        y_true.append(row["score"])

    # regression metrics
    mse = mean_squared_error(y_true, y_scores)
    r2 = r2_score(y_true, y_scores)

    y_pred = [1 if s > 0.3 else -1 if s < -0.3 else 0 for s in y_scores]
    y_true_classes = [int(round(s)) for s in y_true]

    acc = accuracy_score(y_true_classes, y_pred)
    prec = precision_score(y_true_classes, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true_classes, y_pred, average='weighted')
    f1 = f1_score(y_true_classes, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true_classes, y_pred)
    cm = confusion_matrix(y_true_classes, y_pred)

    y_true_bin = label_binarize(y_true_classes, classes=[-1, 0, 1])
    y_score_bin = label_binarize(y_pred, classes=[-1, 0, 1])
    roc_auc = roc_auc_score(y_true_bin, y_score_bin, average='macro', multi_class='ovo')

    print(f"Sentiment Regression Metrics:")
    print(f"- MSE: {mse:.4f}")
    print(f"- R²: {r2:.4f}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall: {rec:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- ROC-AUC: {roc_auc:.4f}")
    print(f"- Cohen's Kappa: {kappa:.4f}")
    print(f"- Confusion Matrix:\n{cm}")


def test(model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TinyFinBERTRegressor()
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()

    texts = [
        "The company's earnings exceeded expectations.",
        "They faced major losses this quarter.",
        "They didn't face major losses this quarter.",
        "Stock prices remained the same.",
        "boost",
        "strong boost",
        "AMD was not able to reduce losses.",
        "AMD reduced debt significantly, improves balance sheet",
        "Economic indicators point to contraction in telecom sector",
        "Company didn't have increased losses over last years."
    ]

    for text in texts:
        clean_text = preprocess_texts([text])[0]
        print(f"Original Text: {text}")
        print(f"Processed Text: {clean_text}")

        tokens = tokenizer.tokenize(clean_text)
        print(f"Tokens: {tokens}")

        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            score = model(**inputs)["score"].item()

        print(f"Predicted Sentiment Score: {score:.3f}")
        sentiment = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
        print(f"Sentiment: {sentiment}\n")


def init_model():
    """Function to properly initialize model with position_ids regardless of whether it's being loaded or created new"""
    model = TinyFinBERTRegressor()

    # Make sure position_ids is registered
    if not hasattr(model.bert.embeddings, 'position_ids'):
        model.bert.embeddings.register_buffer(
            "position_ids",
            torch.arange(512).expand((1, -1)),
            persistent=False,
        )
    return model


def create_api_model(model_path):
    """Create a model suitable for a FastAPI application"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Initialize model with position_ids properly registered
    model = init_model()
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device


if __name__ == "__main__":
    model_dir = "./saved_model"
    phrase_path = "./Sentences_50Agree.txt"
    words_path = "./financial_sentiment_words_phrases_negations.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(os.path.join(model_dir, "pytorch_model.bin")):
        print("Training new model...")
        train_model(phrase_path, words_path, model_dir)
    else:
        print(f"Model found at {os.path.join(model_dir, 'pytorch_model.bin')}")

    evaluate_model(phrase_path, model_dir)
    test(model_dir)