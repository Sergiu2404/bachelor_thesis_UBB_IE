import os
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
from torch.optim import AdamW

RANDOM_SEED = 42
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        original_label = float(self.labels[idx])

        if original_label == 1:
            transformed_label = np.random.uniform(0.8, 1.0)
        else:
            transformed_label = np.random.uniform(0.0, 0.2)

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
            'label': torch.tensor(transformed_label, dtype=torch.float)
        }


class CredibilityRegressor(torch.nn.Module):
    def __init__(self, base_model_name, hidden_dropout_prob=0.1):
        """
        Enhanced TinyBERT model for credibility regression

        Based on papers on regression fine-tuning of transformer models and
        specifically credibility detection architectures
        """
        super(CredibilityRegressor, self).__init__()

        # Load base TinyBERT model
        self.bert = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
        config = self.bert.config
        hidden_size = config.hidden_size  # 312 for TinyBERT 4L

        # 1. Add additional dropout for regularization
        # (helps prevent overfitting on small datasets)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

        # 2. Add pooling mechanism options
        # Literature shows different pooling methods can impact performance
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)

        # 3. Add linguistic feature integration layer
        # Multiple papers show combining BERT with linguistic features improves performance
        self.linguistic_feature_size = 8  # Example: sentiment, readability, etc.

        # 4. Add multi-layer projection head (recommended by regression fine-tuning papers)
        # Multiple non-linear layers help with adapting to regression tasks
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + self.linguistic_feature_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),  # GELU activation (used in BERT) often works better than ReLU
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
        last_hidden_state = outputs.hidden_states[-1]
        cls_output = last_hidden_state[:, 0, :]

        last_hidden_state_permuted = last_hidden_state.permute(0, 2, 1)
        avg_pooled = self.avg_pool(last_hidden_state_permuted).squeeze(-1)

        batch_size = input_ids.shape[0]
        if linguistic_features is None:
            linguistic_features = torch.zeros(batch_size, self.linguistic_feature_size, device=input_ids.device)

        # Concatenate different feature representations
        combined_features = torch.cat([cls_output, avg_pooled, linguistic_features], dim=1)

        # Apply dropout for regularization
        combined_features = self.dropout(combined_features)

        # Pass through regressor layers
        logits = self.regressor(combined_features)

        # Apply sigmoid to get prediction in [0,1] range
        credibility_score = self.sigmoid(logits)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # MSE Loss is recommended for regression tasks in papers
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(credibility_score.view(-1), labels.view(-1))

        return type('ModelOutput', (), {'loss': loss, 'logits': logits, 'credibility_score': credibility_score})


def load_custom_data(custom_file):
    df = pd.read_csv(custom_file)
    required_columns = {"text", "credibility_score"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {required_columns}")
    df = df.dropna(subset=["text", "credibility_score"])
    return df



def load_chunked_data(chunk_files):
    all_data = pd.DataFrame()
    for file in chunk_files:
        df = pd.read_csv(file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    #quarter_index = len(all_data) // 4
    #all_data = all_data.iloc[:quarter_index].reset_index(drop=True)

    train_df, temp_df = train_test_split(all_data, test_size=0.3, random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)
    return train_df, val_df, test_df


def prepare_dataloaders(train_df, val_df, test_df, tokenizer, max_length):
    train_dataset = FinancialNewsDataset(train_df['text'].values, train_df['credibility_score'].values, tokenizer,
                                         max_length)
    val_dataset = FinancialNewsDataset(val_df['text'].values, val_df['credibility_score'].values, tokenizer, max_length)
    test_dataset = FinancialNewsDataset(test_df['text'].values, test_df['credibility_score'].values, tokenizer,
                                        max_length)

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
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                labels=batch['label'])
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
        score = torch.sigmoid(logits).squeeze().item()
    return score


def main():
    output_dir = "E:/saved_models/credibility_analyzer_tinybert"
    os.makedirs(output_dir, exist_ok=True)

    model_file = os.path.join(output_dir, "best_model.pt")
    tokenizer_dir = output_dir

    model_exists = os.path.exists(model_file)

    if model_exists:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForSequenceClassification.from_pretrained(tokenizer_dir)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        chunk_files = [f"./datasets/welfake_chunk_{i}.csv" for i in range(1, 9)]
        #custom_dataset = load_custom_data("./datasets/custom_financial_news_credibility.csv")
        train_df, val_df, test_df = load_chunked_data(chunk_files)

        # train_df = pd.concat([train_df, custom_dataset], ignore_index=True)
        # train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        train_df, temp_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

        train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(train_df, val_df, test_df, tokenizer,
                                                                                MAX_LENGTH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
        model = model.to(DEVICE)

        model = train_model(model, train_dataloader, val_dataloader, EPOCHS, save_path=model_file)

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    example_news_list = [
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

    print("\n--- Credibility Scores ---")
    for i, news in enumerate(example_news_list, 1):
        score = predict_credibility(news, model, tokenizer, MAX_LENGTH)
        print(f"{i:02d}. [{score:.4f}] {news}")


if __name__ == "__main__":
    main()