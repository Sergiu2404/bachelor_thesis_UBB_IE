import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from textblob import TextBlob
import textstat
import spacy

RANDOM_SEED = 42
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

nlp = spacy.load("en_core_web_sm")

def extract_linguistic_features(text):
    blob = TextBlob(text)
    doc = nlp(text)
    return np.array([
        len(text.split()),
        sum(len(word) for word in text.split()) / max(len(text.split()), 1),
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        textstat.flesch_reading_ease(text),
        len([ent for ent in doc.ents]),
        text.lower().count("reportedly"),
        text.lower().count("allegedly"),
    ])

class FinancialNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        original_label = float(self.labels[idx])
        if original_label == 1:
            transformed_label = np.random.uniform(0.8, 1.0)
        else:
            transformed_label = np.random.uniform(0.0, 0.2)
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        features = extract_linguistic_features(text)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(transformed_label, dtype=torch.float),
            'linguistic_features': torch.tensor(features, dtype=torch.float)
        }

class CredibilityRegressor(torch.nn.Module):
    def __init__(self, base_model_name, hidden_dropout_prob=0.1):
        super(CredibilityRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model_name, output_hidden_states=True)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.linguistic_feature_size = 8
        hidden_size = self.bert.config.hidden_size
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + self.linguistic_feature_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(hidden_dropout_prob),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(hidden_dropout_prob),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask, linguistic_features=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        stacked = torch.stack(hidden_states[-4:])
        mean_last_4 = torch.mean(stacked, dim=0)
        cls_output = mean_last_4[:, 0, :]
        if linguistic_features is None:
            batch_size = input_ids.shape[0]
            linguistic_features = torch.zeros(batch_size, self.linguistic_feature_size, device=input_ids.device)
        combined = torch.cat([cls_output, linguistic_features], dim=1)
        combined = self.dropout(combined)
        logits = self.regressor(combined)
        credibility_score = self.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(credibility_score.view(-1), labels.view(-1))
        return type('ModelOutput', (), {'loss': loss, 'logits': logits, 'credibility_score': credibility_score})

def load_chunked_data(chunk_files):
    all_data = pd.DataFrame()
    for file in chunk_files:
        df = pd.read_csv(file)
        all_data = pd.concat([all_data, df], ignore_index=True)
    train_df, temp_df = train_test_split(all_data, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)
    return train_df, val_df, test_df

def prepare_dataloaders(train_df, val_df, test_df, tokenizer, max_length):
    train_dataset = FinancialNewsDataset(train_df['text'].values, train_df['credibility_score'].values, tokenizer, max_length)
    val_dataset = FinancialNewsDataset(val_df['text'].values, val_df['credibility_score'].values, tokenizer, max_length)
    test_dataset = FinancialNewsDataset(test_df['text'].values, test_df['credibility_score'].values, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train_dataloader, val_dataloader, test_dataloader

def train_model(model, train_dataloader, val_dataloader, epochs, save_path):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], linguistic_features=batch['linguistic_features'], labels=batch['label'])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], linguistic_features=batch['linguistic_features'], labels=batch['label'])
                val_loss += outputs.loss.item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
    return model

def predict_credibility(text, model, tokenizer, max_length):
    model.eval()
    encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    features = extract_linguistic_features(text)
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    features = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], linguistic_features=features)
        score = outputs.credibility_score.squeeze().item()
    return score

def main():
    output_dir = "./credibility_deberta"
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, "best_model.pt")
    tokenizer_dir = output_dir
    model_exists = os.path.exists(model_file)
    if model_exists:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = CredibilityRegressor(MODEL_NAME)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        chunk_files = [f"./datasets/welfake_chunk_{i}.csv" for i in range(1, 9)]
        train_df, val_df, test_df = load_chunked_data(chunk_files)
        train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(train_df, val_df, test_df, tokenizer, MAX_LENGTH)
        model = CredibilityRegressor(MODEL_NAME).to(DEVICE)
        model = train_model(model, train_dataloader, val_dataloader, EPOCHS, save_path=model_file)
        model.bert.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    examples = [
        "A economic meltdown is expected due to the financial disater of the last years",
        "Breaking: Tech giant reports record profits, beating analyst expectations by 20%.",
        "Federal Reserve raises interest rates to combat inflation.",
        "Tesla delivers over 400,000 vehicles in Q1, beating Wall Street estimates.",
        "Oil prices rise as OPEC announces production cuts.",
        "U.S. job market adds 300,000 jobs in April, unemployment at 3.6%.",
        "Aliens land on Wall Street and offer stock tips to investors.",
        "Bitcoin declared the official currency of Mars by Elon Musk.",
        "Global economy to be replaced by banana-based trade system.",
        "Scientists find evidence that money grows on trees in Brazil.",
        "Stock market crashes due to mass hysteria over TikTok dance."
    ]
    for news in examples:
        score = predict_credibility(news, model, tokenizer, MAX_LENGTH)
        print(f"[{score:.4f}] {news}")

if __name__ == "__main__":
    main()
