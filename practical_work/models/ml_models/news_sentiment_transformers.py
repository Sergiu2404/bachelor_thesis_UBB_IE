import re

import torch
import io
import pandas as pd
from transformers import BertTokenizer

# Force PyTorch to use CPU
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

import pandas as pd
import re
import kagglehub

# Load Financial PhraseBank dataset
def load_financial_phrasebank():
    """Load and preprocess the Financial PhraseBank dataset."""
    print("Loading Financial PhraseBank dataset...")
    data = []
    with open("fake_news_datasets/FinancialPhraseBank-v1.0/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r",
              encoding="ISO-8859-1") as file:
        for line in file:
            match = re.search(r"@(neutral|positive|negative)\s*$", line.strip())
            if match:
                sentiment = match.group(1)
                text = line[:match.start()].strip()
                label = {"neutral": 0, "positive": 1, "negative": 2}[sentiment]
                data.append({"text": text, "sentiment": label})
            else:
                print(f"Skipping malformed line: {line.strip()}")

    df = pd.DataFrame(data)
    print("Financial PhraseBank Loaded:", df.shape)
    return df

df = load_financial_phrasebank()

# Load FiQA dataset
splits = {
    'train': 'data/train-00000-of-00001-aeefa1eadf5be10b.parquet',
    'test': 'data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet',
    'valid': 'data/valid-00000-of-00001-51867fe1ac59af78.parquet'
}

df_fiqa = pd.read_parquet("hf://datasets/TheFinAI/fiqa-sentiment-classification/" + splits["train"])

# relevant columns
df_fiqa = df_fiqa[['sentence', 'score']].rename(columns={'sentence': 'text', 'score': 'sentiment'})

def convert_score_to_label(score):
    if score < -0.05:
        return 0  # Neutral
    elif score > 0.05:
        return 2  # Positive
    else:
        return 1  # Negative

df_fiqa['sentiment'] = df_fiqa['sentiment'].apply(convert_score_to_label)
df_combined = pd.concat([df, df_fiqa], ignore_index=True)
print("After merging PhraseBank and FiQA:", df_combined.shape)

#  Kaggle dataset Sentiment Analysis for Financial News
path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
kaggle_df = pd.read_csv(f"{path}/all-data.csv", encoding="ISO-8859-1", header=None)

# Rename columns to match existing datasets
kaggle_df.columns = ["sentiment", "text"]
sentiment_mapping = {"negative": 2, "neutral": 0, "positive": 1}
kaggle_df["sentiment"] = kaggle_df["sentiment"].map(sentiment_mapping)

df_combined = pd.concat([df_combined, kaggle_df], ignore_index=True)
print("Final dataset shape:", df_combined.shape)
print(df_combined['sentiment'].value_counts())
df_combined.tail(20)

contents = df.text.values
labels = df.sentiment.values




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 0

for sentence in contents:
    # encode each sentence and transform to tensor for padding later regarding longest sentence
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True))
    max_len = max(max_len, len(input_ids))

print(max_len)



input_ids = []
attention_masks = []

for sent in contents:
    encoded_dict = tokenizer.encode_plus(
                        sent,
                        add_special_tokens = True, # add [CLS] and [SEP]
                        truncation=True,           # cut longer sent to max_length
                        max_length = 64,
                        padding='max_length',
                        pad_to_max_length = True,  # to ensure all input has same length
                        return_attention_mask = True,   # construct attn masks for excluding padding tokens
                        return_tensors = 'pt',     # return as pytorch tensors for using gpu power
                   )

    input_ids.append(encoded_dict['input_ids']) # tokenized sentence stacked together
    attention_masks.append(encoded_dict['attention_mask']) # arr of 1 for sentence token or 0 for padding token

input_ids = torch.cat(input_ids, dim=0) # concats all sentences together
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

#sentence 0 as list of ids
print('Original: ', contents[0])
print('Token IDs:', input_ids[0])



from torch.utils.data import TensorDataset, random_split

# for using all these in a dataloader
dataset = TensorDataset(input_ids, attention_masks, labels)

# train, validation (for fine-tuning), test
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# dataset sizes
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

# split dataset randomly according to sizes
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Print dataset sizes
print(f"Train samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Test samples: {test_size}")




from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# for fine-tuning bert on a specific task, the authors recommend a batch size of 16 or 32.
batch_size = 32

# create DataLoaders for training, validation and testing sets.
train_dataloader = DataLoader(
    train_dataset,  # training samples
    sampler = RandomSampler(train_dataset),  # get batches randomly
    batch_size = batch_size  # trains with this batch size.
)

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset), # get batches sequentially.
    batch_size = batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)



from transformers import BertForSequenceClassification, AdamW, BertConfig

# load BertForSequenceClassification pretrained BERT model with a linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # use BERT model with uncased vocab.
    num_labels = 3,  # 3 output labels for classification
    output_attentions = False,  # dont need attentions weights.
    output_hidden_states = False,  # dont need hidden-states.
    return_dict = False
)

model.train()
# run on GPU
model.cuda()


params = list(model.named_parameters())
# print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('\nemb Layer\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n1st Transformer\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\noutput\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# AdamW optimizer for reducing overfitting (compromising accuracy)
optimizer = AdamW(
    model.parameters(),
    lr = 2e-5, # args.learning_rate - default is 5e-5
    eps = 1e-8
)





from transformers import get_linear_schedule_with_warmup


# recommended 2 - 4 passes through entire dataset
epochs = 4

# training steps: [nr batches] x [nr epochs]
total_steps = len(train_dataloader) * epochs

# scheduler modifies learning rate during training
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    # no warmup => starts from high learning rate for exploring faster at beginning and gradually decreases it
    num_training_steps = total_steps
)



# training loop
import time
import datetime
import random
import numpy as np


# calculate accuracy of predictions for given labels
def flat_accuracy(predictions, labels):
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Take time in seconds and return string hh:mm:ss
    '''
    # round to seconds
    elapsed_rounded = int(round((elapsed)))
    # format -> hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# set fixed seed to ensure random operations in libraries generate same results
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

training_stats = []
# measure training time for the whole run
total_t0 = time.time()


# iterate over entire dataset epochs times
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # measure training time
    t0 = time.time()
    # calculate loss for each epoch
    total_train_loss = 0
    # run training
    model.train()

    # iterate over training batches
    for step, batch in enumerate(train_dataloader):
        # progress every 40 batches
        if step % 40 == 0 and not step == 0:
            # elapsed time in minutes
            elapsed = format_time(time.time() - t0)
            # print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # unpack training batch from dataloader
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # clear gradients accumulates in prev steps before performing backward pass
        model.zero_grad()

        # forward pass (evaluate the model on this training batch), get output from bert classification model
        outputs = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels
        )
        loss = outputs[0]
        logits = outputs[1]  # raw output predictions before applying activation fn

        # print("\/\/\/\/\/\/\/")
        # print("Loss:", loss)
        # print("Logits shape:", logits.shape)
        # print("Labels shape:", b_labels.shape)
        # print("Unique labels:", torch.unique(b_labels))

        # sum up training loss over all batches for getting average loss, loss is a Tensor containing the actual loss, backpropagate to adjust weights
        total_train_loss += loss.item()
        loss.backward()
        # hold norm of gradient under 1 for preventing exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters based on calculated gradients, adjust weights to minimize loss
        optimizer.step()
        # update learning rate
        scheduler.step()

    # calc average loss of all batches
    avg_train_loss = total_train_loss / len(train_dataloader)
    # measure time for current epoch
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # eval data for one epoch
    for batch in validation_dataloader:
        # unpack validation batches
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # no need for gradient computation in validation / testing phase, only in training
        with torch.no_grad():
            # forward pass, calculate logit predictions
            # calculate logit predictions
            (loss, logits) = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )

        # accumulate validation loss
        total_eval_loss += loss.item()

        # move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # calculate accuracy for current batch and accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # average loss over all batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # measure validation time
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # record all stats for current epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))




print("\nRun Test Evaluation...")

t0 = time.time()
model.eval()

total_test_accuracy = 0
total_test_loss = 0

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    # no gradient computation needed since testing phase
    with torch.no_grad():
        (loss, logits) = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels
        )

    total_test_loss += loss.item()
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    total_test_accuracy += flat_accuracy(logits, label_ids)

avg_test_accuracy = total_test_accuracy / len(test_dataloader)
avg_test_loss = total_test_loss / len(test_dataloader)

print("  Test Accuracy: {0:.2f}".format(avg_test_accuracy))
print("  Test Loss: {0:.2f}".format(avg_test_loss))
print("  Test Evaluation took: {:}".format(format_time(time.time() - t0)))



import torch

model.eval()
def preprocess_text(text, tokenizer, max_length=512):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,    # add special tokens like [CLS] and [SEP]
        max_length=max_length,      # max len of tokenized sequence
        padding='max_length',       # Pad to max_length
        truncation=True,            # truncate if longer than max_length
        return_tensors='pt',        # return as PyTorch tensor
    )
    return encoding


#text = "The stock market saw a significant drop today after disappointing earnings reports from major tech companies." negative
text = "The stock market kept the same track for last 3 months."
#text = "Tesla has success after making announcements about their new electric vehicle" positive
inputs = preprocess_text(text, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():  # no gradient computation needed since using the model only
    outputs = model(**inputs)

# raw prediction scores
logits = outputs[0]
# Convert logits to probabilities using softmax
probabilities = torch.nn.functional.softmax(logits, dim=-1)
# Get predicted sentiment class
predicted_class = torch.argmax(probabilities, dim=-1).item()

print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities}")
sentiment_classes = ["neutral", "positive", "negative"]
print(f"Predicted sentiment: {sentiment_classes[predicted_class]}")