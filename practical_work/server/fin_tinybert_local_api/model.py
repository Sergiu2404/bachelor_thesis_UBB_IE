from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

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




from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # tanh to limit output between -1 and 1
        score = torch.tanh(self.regressor(pooled_output)).squeeze()

        loss = F.mse_loss(score, labels) if labels is not None else None
        return {'loss': loss, 'score': score}
