import os
import re
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from nltk.corpus import stopwords
from datasets import Dataset
from sklearn.model_selection import train_test_split
import spacy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class TinyFinBERTRegressor(nn.Module):
    def __init__(self, pretrained_model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model, config=self.config)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        weights = self.attention(hidden_states)
        weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        weights = torch.softmax(weights, dim=1)
        pooled_output = (hidden_states * weights).sum(dim=1)
        score = self.regressor(pooled_output).squeeze()
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
            if token.lemma_.strip()
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

def train_model(train_df, test_df, save_path, extra_df=None):
    os.makedirs(save_path, exist_ok=True)
    train_df['text'] = preprocess_texts(train_df['text'])
    if extra_df is not None:
        extra_df['text'] = preprocess_texts(extra_df['text'])
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
    test_df['text'] = preprocess_texts(test_df['text'])
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    def tokenize(batch):
        tokens = tokenizer(batch["text"], padding='max_length', truncation=True, max_length=128)
        tokens["labels"] = batch["score"]
        return tokens
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    args = TrainingArguments(
        output_dir=os.path.join(save_path, "results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    model = TinyFinBERTRegressor().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
    torch.save(model.state_dict(), os.path.join(save_path, "regressor_model.pt"))
    tokenizer.save_pretrained(save_path)

def evaluate_base_tinybert_on_phrasebank(test_df):
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyFinBERTRegressor().to(device)  # randomly initialized regressor

    model.eval()
    test_df['text'] = preprocess_texts(test_df['text'])

    y_true, y_pred = [], []

    for _, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            score = model(**inputs)["score"].item()
        y_pred.append(score)
        y_true.append(row["score"])

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_pred_class = [1 if s > 0.3 else -1 if s < -0.3 else 0 for s in y_pred]
    y_true_class = [int(round(s)) for s in y_true]

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, average='weighted')
    f1 = f1_score(y_true_class, y_pred_class, average='weighted')
    kappa = cohen_kappa_score(y_true_class, y_pred_class)
    cm = confusion_matrix(y_true_class, y_pred_class)

    y_true_bin = label_binarize(y_true_class, classes=[-1, 0, 1])
    y_pred_bin = label_binarize(y_pred_class, classes=[-1, 0, 1])
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovo')

    print("\n=== Evaluation: Base TinyBERT (No Fine-tuning) ===")
    print(f"- MSE: {mse:.4f}")
    print(f"- R²: {r2:.4f}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall: {rec:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- ROC-AUC: {roc_auc:.4f}")
    print(f"- Cohen's Kappa: {kappa:.4f}")
    print(f"- Confusion Matrix:\n{cm}")


from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

def predict_sentiments_on_news(news_list, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyFinBERTRegressor().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "regressor_model.pt"), map_location=device))
    model.eval()

    processed_texts = preprocess_texts(news_list)
    predictions = []

    for text in processed_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            score = model(**inputs)["score"].item()
        label = "Positive" if score > 0.3 else "Negative" if score < -0.3 else "Neutral"
        predictions.append((text, score, label))

    return predictions

def evaluate_fine_tuned_tinybert_on_phrasebank(test_df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyFinBERTRegressor().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "regressor_model.pt"), map_location=device))
    model.eval()

    test_df['text'] = preprocess_texts(test_df['text'])
    y_true, y_pred = [], []

    for _, row in test_df.iterrows():
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, padding='max_length', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            score = model(**inputs)["score"].item()
        y_pred.append(score)
        y_true.append(row["score"])

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    y_pred_class = [1 if s > 0.3 else -1 if s < -0.3 else 0 for s in y_pred]
    y_true_class = [int(round(s)) for s in y_true]

    acc = accuracy_score(y_true_class, y_pred_class)
    prec = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, average='weighted')
    f1 = f1_score(y_true_class, y_pred_class, average='weighted')
    kappa = cohen_kappa_score(y_true_class, y_pred_class)
    cm = confusion_matrix(y_true_class, y_pred_class)

    y_true_bin = label_binarize(y_true_class, classes=[-1, 0, 1])
    y_pred_bin = label_binarize(y_pred_class, classes=[-1, 0, 1])
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin, average='macro', multi_class='ovo')

    print("\n=== Evaluation: Fine-Tuned TinyBERT ===")
    print(f"- MSE: {mse:.4f}")
    print(f"- R²: {r2:.4f}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall: {rec:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- ROC-AUC: {roc_auc:.4f}")
    print(f"- Cohen's Kappa: {kappa:.4f}")
    print(f"- Confusion Matrix:\n{cm}")


from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    phrase_path = "Sentences_50Agree.txt"
    words_path = "financial_sentiment_words_phrases_negations.csv"
    model_dir = "E:/saved_models/attention_fine_tuned_tinybert"
    phrase_df = load_phrasebank(phrase_path)
    train_phrase, test_phrase = train_test_split(phrase_df, test_size=0.15, random_state=42)
    words_df = load_words_phrases(words_path)
    if not os.path.isfile(os.path.join(model_dir, "regressor_model.pt")):
        train_model(train_phrase.copy(), test_phrase.copy(), model_dir, extra_df=words_df.copy())

    news_samples = [
        "Federal Reserve raises interest rates again.",
        "Apple stock plunges after poor earnings report.",
        "The company maintained stable growth throughout the year."
    ]

    results = predict_sentiments_on_news(news_samples, model_dir)
    for i, (text, score, label) in enumerate(results):
        print(f"\nNews {i + 1}: {text}")
        print(f" - Sentiment Score: {score:.4f}")
        print(f" - Predicted Label: {label}")

    evaluate_base_tinybert_on_phrasebank(test_phrase.copy())
    evaluate_fine_tuned_tinybert_on_phrasebank(test_phrase.copy(), model_dir)

#
# === Evaluation: Base TinyBERT (No Fine-tuning) ===
# - MSE: 0.3666
# - R²: 0.0284
# - Accuracy: 0.5247
# - Precision: 0.4149
# - Recall: 0.5247
# - F1 Score: 0.4486
# - ROC-AUC: 0.4845
# - Cohen's Kappa: -0.0484
# - Confusion Matrix:
# [[  0 108   2]
#  [  0 471 100]
#  [  0 251  38]]





# === Evaluation: Fine-Tuned TinyBERT with self attention layer ===
# class TinyFinBERTRegressor(nn.Module):
#     def __init__(self, pretrained_model='huawei-noah/TinyBERT_General_4L_312D'):
#         super().__init__()
#         self.config = AutoConfig.from_pretrained(pretrained_model)
#         self.bert = AutoModel.from_pretrained(pretrained_model, config=self.config)
#         self.attention = nn.Sequential(
#             nn.Linear(self.config.hidden_size, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1)
#         )
#         self.regressor = nn.Linear(self.config.hidden_size, 1)
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.last_hidden_state
#         weights = self.attention(hidden_states)
#         weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
#         weights = torch.softmax(weights, dim=1)
#         pooled_output = (hidden_states * weights).sum(dim=1)
#         score = self.regressor(pooled_output).squeeze()
#         loss = F.mse_loss(score, labels) if labels is not None else None
#         return {'loss': loss, 'score': score}
# - MSE: 0.1668
# - R²: 0.5607
# - Accuracy: 0.8033
# - Precision: 0.8117
# - Recall: 0.8033
# - F1 Score: 0.8053
# - ROC-AUC: 0.8474
# - Cohen's Kappa: 0.6564
# - Confusion Matrix:
# [[ 68  10   6]
#  [ 31 341  55]
#  [  2  39 175]]