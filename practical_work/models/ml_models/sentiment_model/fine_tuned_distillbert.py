# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from datasets import Dataset
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import nltk
# from nltk.corpus import stopwords
# import spacy
# import os
#
# nlp = spacy.load("en_core_web_sm")
#
# nltk.download('stopwords', quiet=True)
#
# negation_words = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'scarcely',
#                   'barely', "n't", "without", "unless", "nor"}
# stop_words = set(stopwords.words('english')) - negation_words
#
#
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     words = text.split()
#     filtered_words = [word for word in words if word not in stop_words]
#     return ' '.join(filtered_words)
#
#
# def load_financial_phrasebank(file_path):
#     with open(file_path, 'r', encoding='latin1') as f:
#         lines = f.readlines()
#     sentences = []
#     labels = []
#     for line in lines:
#         if '@' in line:
#             parts = line.strip().split('@')
#             if len(parts) == 2:
#                 sentence, label = parts
#                 sentences.append(preprocess_text(sentence))
#                 if label.lower() == 'positive':
#                     labels.append(2)
#                 elif label.lower() == 'neutral':
#                     labels.append(1)
#                 elif label.lower() == 'negative':
#                     labels.append(0)
#     return pd.DataFrame({'text': sentences, 'label': labels})
#
#
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
#
#
# def predict_sentiment(text, model, tokenizer, device):
#     text = preprocess_text(text)
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
#     model = model.to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probabilities = F.softmax(logits, dim=1)
#     sentiment_scores = torch.tensor([-1.0, 0.0, 1.0], device=device)
#     weighted_score = torch.sum(probabilities * sentiment_scores)
#     return weighted_score.item()
#
#
# def train_model(dataset_path, model_save_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     print("Loading and preprocessing dataset...")
#     df = load_financial_phrasebank(dataset_path)
#     print(f"Dataset loaded with {len(df)} samples")
#
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
#     print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
#
#     train_dataset = Dataset.from_pandas(train_df)
#     val_dataset = Dataset.from_pandas(val_df)
#
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#
#     def tokenize_function(examples):
#         processed_texts = [" ".join([token.text for token in nlp(preprocess_text(t))]) for t in examples["text"]]
#         return tokenizer(processed_texts, padding='max_length', truncation=True, max_length=128)
#
#     print("Tokenizing datasets...")
#     train_dataset = train_dataset.map(tokenize_function, batched=True)
#     val_dataset = val_dataset.map(tokenize_function, batched=True)
#
#     os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
#     results_dir = os.path.join(os.path.dirname(model_save_path), "results")
#     logs_dir = os.path.join(os.path.dirname(model_save_path), "logs")
#     os.makedirs(results_dir, exist_ok=True)
#     os.makedirs(logs_dir, exist_ok=True)
#
#     training_args = TrainingArguments(
#         output_dir=results_dir,
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=64,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir=logs_dir,
#         logging_steps=10,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model='f1',
#     )
#
#     model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics,
#     )
#
#     print("Starting training...")
#     trainer.train()
#
#     print(f"Saving model to {model_save_path}")
#     trainer.save_model(model_save_path)
#     tokenizer.save_pretrained(model_save_path)
#     print("Model saved successfully")
#
#     return model, tokenizer
#
#
# def load_model(model_path, device):
#     print(f"Loading model from {model_path}")
#     model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
#     tokenizer = DistilBertTokenizer.from_pretrained(model_path)
#     return model, tokenizer
#
#
# def test_model(model, tokenizer, device):
#     examples = [
#         "The company's quarterly earnings exceeded expectations, leading to a surge in stock prices.",
#         "The company faced the same increasing losses and unemployment as the same period last year.",
#         "The company's stock prices kept the same trend for the last months."
#     ]
#
#     for text in examples:
#         score = predict_sentiment(text, model, tokenizer, device)
#         print(f"Text: {text}")
#         print(f"Sentiment score: {score:.3f}")
#         if score > 0.3:
#             print("Prediction: Positive")
#         elif score < -0.3:
#             print("Prediction: Negative")
#         else:
#             print("Prediction: Neutral")
#         print("-" * 50)
#
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dataset_path = "../fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt"
#     model_path = "E:/saved_models/sentiment_analysis_fine_tuned_distillbert"
#
#     model_files_exist = os.path.isfile(f"{model_path}/pytorch_model.bin") or \
#                         os.path.isfile(f"{model_path}/model.safetensors")
#
#     if model_files_exist:
#         model, tokenizer = load_model(model_path, device)
#     else:
#         model, tokenizer = train_model(dataset_path, model_path)
#
#     param_size = 0
#     buffer_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#
#     size_all_mb = (param_size + buffer_size) / 1024 ** 2
#     print('Size: {:.3f} MB'.format(size_all_mb))
#
#     test_model(model, tokenizer, device)
#
#
# if __name__ == "__main__":
#     main()







from transformers import DistilBertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import spacy
from nltk.corpus import stopwords
import os


class DistilFinBERT(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased', num_labels=3):
        super(DistilFinBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, 256)
        self.classifier = nn.Linear(256, num_labels)
        self.sentiment_weights = torch.tensor([-1.0, 0.0, 1.0])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = self.dropout(hidden_state)
        intermediate = F.relu(self.pre_classifier(hidden_state))
        intermediate = self.dropout(intermediate)
        logits = self.classifier(intermediate)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    def predict_sentiment_score(self, logits):
        probs = F.softmax(logits, dim=1)
        sentiment_weights = self.sentiment_weights.to(probs.device)
        score = torch.sum(probs * sentiment_weights.unsqueeze(0), dim=1)
        return score


class FinancialSentimentAnalyzer:
    def __init__(self, model_path=None, device=None):
        self.nlp = spacy.load("en_core_web_sm")

        negation_words = {'no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
                          'hardly', 'scarcely', 'barely', "n't", "without", "unless", "nor"}
        self.stop_words = set(stopwords.words('english')) - negation_words

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        from transformers import DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            model_path if model_path and os.path.exists(model_path) else 'distilbert-base-uncased'
        )

        self.model = DistilFinBERT().to(self.device)

        param_size = 0
        buffer_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Size: {:.3f} MB'.format(size_all_mb))

        if model_path and os.path.exists(model_path):
            model_file = os.path.join(model_path, "model.pt")
            if os.path.isfile(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                print(f"Model loaded from {model_file}")
            else:
                print(f"Model file {model_file} not found. Using untrained model.")

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def predict(self, text):
        processed_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            sentiment_score = self.model.predict_sentiment_score(logits)

        score = sentiment_score.item()
        sentiment_label = "Positive" if score > 0.3 else "Negative" if score < -0.3 else "Neutral"

        return {
            "score": score,
            "sentiment": sentiment_label,
            "probabilities": F.softmax(logits, dim=1).cpu().numpy()[0]
        }


def train_distilfinbert(dataset_path, model_save_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_financial_phrasebank(file_path):
        with open(file_path, 'r', encoding='latin1') as f:
            lines = f.readlines()
        sentences = []
        labels = []
        analyzer = FinancialSentimentAnalyzer(device=device)
        for line in lines:
            if '@' in line:
                parts = line.strip().split('@')
                if len(parts) == 2:
                    sentence, label = parts
                    sentences.append(analyzer.preprocess_text(sentence))
                    if label.lower() == 'positive':
                        labels.append(2)
                    elif label.lower() == 'neutral':
                        labels.append(1)
                    elif label.lower() == 'negative':
                        labels.append(0)
        return pd.DataFrame({'text': sentences, 'label': labels})

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    os.makedirs(model_save_path, exist_ok=True)
    results_dir = os.path.join(model_save_path, "results")
    logs_dir = os.path.join(model_save_path, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Loading dataset from {dataset_path}")
    df = load_financial_phrasebank(dataset_path)
    print(f"Dataset loaded with {len(df)} examples")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Train set: {len(train_df)} examples, Validation set: {len(val_df)} examples")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    analyzer = FinancialSentimentAnalyzer(device=device)
    tokenizer = analyzer.tokenizer

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=128)

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    print("Tokenization complete")

    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=logs_dir,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    model = DistilFinBERT().to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete")

    model_file = os.path.join(model_save_path, "model.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

    tokenizer.save_pretrained(model_save_path)
    print(f"Tokenizer saved to {model_save_path}")

    return model, tokenizer


def test_examples(model_path=None):
    print(f"Testing examples using model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    analyzer = FinancialSentimentAnalyzer(model_path=model_path, device=device)

    examples = [
        "The company's quarterly earnings exceeded expectations, leading to a surge in stock prices.",
        "The company faced increasing losses and unemployment compared to the same period last year.",
        "The company's stock prices remained within the same range over the last quarter."
    ]

    for text in examples:
        result = analyzer.predict(text)
        print(f"Text: {text}")
        print(f"Sentiment score: {result['score']:.3f}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Probabilities: Negative={result['probabilities'][0]:.3f}, "
              f"Neutral={result['probabilities'][1]:.3f}, "
              f"Positive={result['probabilities'][2]:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    model_path = "E:\saved_models\sentiment_analysis_fine_tuned_distillbert"
    dataset_path = "../fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

    model_file = os.path.join(model_path, "model.pt")
    model_exists = os.path.isfile(model_file)

    if not model_exists:
        print(f"Model file {model_file} not found. Training a new model...")
        train_distilfinbert(dataset_path, model_path)
    else:
        print(f"Model file {model_file} found. Skipping training.")

    test_examples(model_path)
