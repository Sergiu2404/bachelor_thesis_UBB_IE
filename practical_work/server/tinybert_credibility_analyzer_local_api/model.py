import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

class CredibilityRegressor(nn.Module):
    def __init__(self, pretrained_model='huawei-noah/TinyBERT_General_4L_312D'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.bert = AutoModel.from_pretrained(pretrained_model, config=self.config)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.sensationalism_features = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        self.attention_weights = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.linguistic_patterns = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        self.emotional_intensity = nn.Sequential(
            nn.Linear(self.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(64 + 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.credibility_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        attn_output, _ = self.multi_head_attention(hidden_states, hidden_states, hidden_states,
                                                   key_padding_mask=~attention_mask.bool())

        weights = self.attention_weights(attn_output)
        weights = weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        weights = torch.softmax(weights, dim=1)
        pooled_output = (attn_output * weights).sum(dim=1)

        cls_token = attn_output[:, 0, :]

        sensationalism_feat = self.sensationalism_features(pooled_output)
        linguistic_feat = self.linguistic_patterns(cls_token)
        emotional_feat = self.emotional_intensity(pooled_output)

        combined_features = torch.cat([sensationalism_feat, linguistic_feat, emotional_feat], dim=1)
        fused_features = self.feature_fusion(combined_features)

        score = self.credibility_head(fused_features).squeeze()

        loss = None
        if labels is not None:
            mse_loss = F.mse_loss(score, labels)
            huber_loss = F.smooth_l1_loss(score, labels)
            loss = 0.7 * mse_loss + 0.3 * huber_loss

        return {'loss': loss, 'score': score}

import os

MODEL_DIR = r"E:\saved_models\credibility_regressor_tinybert"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = CredibilityRegressor().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "regressor_model.pt"), map_location=DEVICE))
model.eval()

def predict_credibility(text: str) -> float:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        score = model(**encoded)["score"].item()
    return round(score, 3)
