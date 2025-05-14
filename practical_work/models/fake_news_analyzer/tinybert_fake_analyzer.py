import os
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.optim import AdamW

# Configuration
RANDOM_SEED = 42
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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
        label = float(self.labels[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path):
    df = pd.read_csv(file_path)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
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
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Training loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), save_path)

    return model


def predict_credibility(text, model, tokenizer, max_length):
    model.eval()
    encoding = tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        if logits.shape[1] == 1:
            score = torch.sigmoid(logits).squeeze().item()
        else:
            probs = torch.softmax(logits, dim=1)
            score = probs[0, 1].item()
    return score


def main():
    output_dir = "E:/saved_models/credibility_analyzer_tinybert"
    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(output_dir, "best_model.pt")
    tokenizer_dir = output_dir  # Same dir as model

    model_exists = os.path.exists(model_file)

    if model_exists:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForSequenceClassification.from_pretrained(tokenizer_dir)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        file_path = "./datasets/36k_welfake_dataset.csv"
        train_df, val_df, test_df = load_data(file_path)
        train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(train_df, val_df, test_df, tokenizer, MAX_LENGTH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model = model.to(DEVICE)

        model = train_model(model, train_dataloader, val_dataloader, EPOCHS, save_path=model_file)

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # 10 Example News Headlines
    example_news_list = [
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

    print("\n--- Credibility Scores ---")
    for i, news in enumerate(example_news_list, 1):
        score = predict_credibility(news, model, tokenizer, MAX_LENGTH)
        print(f"{i:02d}. [{score:.4f}] {news}")


if __name__ == "__main__":
    main()
