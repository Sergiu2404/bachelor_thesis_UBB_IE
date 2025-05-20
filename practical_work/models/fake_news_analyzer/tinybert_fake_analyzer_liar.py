import kagglehub
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("doanquanvietnamca/liar-dataset")
csv_path = os.path.join(path, "train.tsv")
df = pd.read_csv(csv_path, sep='\t', header=None)

df = df[[1, 2]]
df.columns = ['label', 'statement']

label_map = {
    "pants-fire": 0.0,
    "false": 0.2,
    "barely-true": 0.4,
    "half-true": 0.6,
    "mostly-true": 0.8,
    "true": 1.0
}
df = df[df['label'].isin(label_map)]
df['credibility_score'] = df['label'].map(label_map)
df = df[['statement', 'credibility_score']]

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

def tokenize(example):
    tokens = tokenizer(
        example["statement"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    return {**tokens, "label": example["credibility_score"]}

raw_dataset = Dataset.from_pandas(df)
dataset = raw_dataset.map(tokenize)

train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
val_dataset = train_test["test"]

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

class CredibilityRegressor(nn.Module):
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        weights = self.attention(hidden_states)
        weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        weights = torch.softmax(weights, dim=1)
        pooled_output = (hidden_states * weights).sum(dim=1)
        score = self.sigmoid(self.regressor(pooled_output)).squeeze()
        loss = F.mse_loss(score, labels) if labels is not None else None
        return {'loss': loss, 'score': score}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CredibilityRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
total_val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_val_loss += outputs["loss"].item()
print(f"Validation Loss: {total_val_loss / len(val_loader):.4f}")

def predict(texts):
    model.eval()
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        #scores = model(**encoded)["score"]
        scores = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                       token_type_ids=encoded.get('token_type_ids'))["score"]

    return scores.cpu().numpy().tolist()

sample_texts = [
    "The moon landing was staged in a Hollywood studio.",
    "Scientists have confirmed water on Mars.",
    "The COVID-19 vaccine contains microchips."
]
print("Predictions:", predict(sample_texts))
