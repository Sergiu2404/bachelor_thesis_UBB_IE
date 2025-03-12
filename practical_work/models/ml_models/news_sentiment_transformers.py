import re
import os
import time
import random
import datetime
import numpy as np
import pandas as pd
import torch
import io
import kagglehub
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from news_negation_handling import apply_negation_handling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


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


def load_fiqa_dataset():
    """Load and preprocess the FiQA dataset."""
    print("Loading FiQA dataset...")
    splits = {
        'train': 'data/train-00000-of-00001-aeefa1eadf5be10b.parquet',
        'test': 'data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet',
        'valid': 'data/valid-00000-of-00001-51867fe1ac59af78.parquet'
    }

    df_fiqa = pd.read_parquet("hf://datasets/TheFinAI/fiqa-sentiment-classification/" + splits["train"])

    df_fiqa = df_fiqa[['sentence', 'score']].rename(columns={'sentence': 'text', 'score': 'sentiment'})

    def convert_score_to_label(score):
        if score < 0:
            return 2
        elif score > 0:
            return 1
        else:
            return 0

    df_fiqa['sentiment'] = df_fiqa['sentiment'].apply(convert_score_to_label)
    print("FiQA dataset loaded:", df_fiqa.shape)
    return df_fiqa


def load_kaggle_dataset():
    """Load and preprocess the Kaggle Sentiment Analysis for Financial News dataset."""
    print("Loading Kaggle Financial News dataset...")
    path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    kaggle_df = pd.read_csv(f"{path}/all-data.csv", encoding="ISO-8859-1", header=None)

    kaggle_df.columns = ["sentiment", "text"]
    sentiment_mapping = {"negative": 2, "neutral": 0, "positive": 1}
    kaggle_df["sentiment"] = kaggle_df["sentiment"].map(sentiment_mapping)

    print("Kaggle dataset loaded:", kaggle_df.shape)
    return kaggle_df


def load_all_datasets():
    """Load and combine all datasets."""
    #df_phrasebank = load_financial_phrasebank()
    df_fiqa = load_fiqa_dataset()
    df_kaggle = load_kaggle_dataset()

    # merge datasets
    df_combined = pd.concat([df_fiqa, df_kaggle], ignore_index=True)
    # print("Final combined dataset shape:", df_combined.shape)
    print("Sentiment class distribution:", df_combined['sentiment'].value_counts())
    return df_combined


def tokenize_data(df, max_length=64):
    """Tokenize the text data using BERT tokenizer."""
    print("Tokenizing data...")
    contents = df.text.values
    labels = df.sentiment.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    attention_masks = []

    for sent in contents:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            truncation=True,  # Cut longer sentences to max_length
            max_length=max_length,
            padding='max_length',  # Pad to max_length
            return_attention_mask=True,  # Construct attention masks
            return_tensors='pt',  # Return PyTorch tensors
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    print(f"Tokenization complete. Input shape: {input_ids.shape}")
    return input_ids, attention_masks, labels, tokenizer


def prepare_dataloaders(input_ids, attention_masks, labels, batch_size=32):
    """Prepare train, validation, and test dataloaders."""
    print("Preparing dataloaders...")

    # create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # split dataset
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    print(f"Train samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")

    return train_dataloader, validation_dataloader, test_dataloader


def initialize_model(num_labels=3):
    """Initialize the BERT model for sequence classification."""
    print("Initializing BERT model...")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False
    )

    # move model to device
    model = model.to(device)

    return model


def initialize_optimizer(model, train_dataloader, epochs=4):
    """Initialize optimizer and learning rate scheduler."""
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)


    # total number of training steps
    total_steps = len(train_dataloader) * epochs

    # create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    return optimizer, scheduler


def flat_accuracy(predictions, labels):
    """Calculate accuracy of predictions for given labels."""
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Take time in seconds and return string hh:mm:ss."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_model(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs=4):
    """Train the model and validate after each epoch."""
    print("Beginning training...")

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print("")
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            loss = outputs[0]
            logits = outputs[1]

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                (loss, logits) = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        # record stats
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    print("")
    print("Training complete!")
    print(f"Total training took {format_time(time.time() - total_t0)}")

    return model, training_stats


def evaluate_model(model, test_dataloader):
    """Evaluate the model on the test dataset."""
    print("\nRunning Test Evaluation...")

    t0 = time.time()
    model.eval()

    total_test_accuracy = 0
    total_test_loss = 0

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

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

    print(f"  Test Accuracy: {avg_test_accuracy:.2f}")
    print(f"  Test Loss: {avg_test_loss:.2f}")
    print(f"  Test Evaluation took: {format_time(time.time() - t0)}")

    return avg_test_accuracy, avg_test_loss


def save_model(model, tokenizer, save_path="E:\\saved_models\\sentiment_analysis_transformer"):
    """Save the model and tokenizer to the specified path."""
    print(f"Saving model to {save_path}...")

    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)

    tokenizer.save_pretrained(save_path)

    print("Model and tokenizer saved successfully!")


def load_model(load_path="E:\\saved_models\\sentiment_analysis_transformer"):
    """Load the model and tokenizer from the specified path."""
    print(f"Loading model from {load_path}...")

    if not os.path.exists(load_path):
        print(f"Model path {load_path} does not exist.")
        return None, None

    try:
        model = BertForSequenceClassification.from_pretrained(load_path)
        model = model.to(device)

        tokenizer = BertTokenizer.from_pretrained(load_path)

        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def predict_sentiment(text, model, tokenizer, max_length=64):
    """Predict sentiment for a given text."""
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    # move inputs to device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    # get predictions, no need for gradient computaiton since not training
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    sentiment_classes = ["neutral", "positive", "negative"]

    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted sentiment: {sentiment_classes[predicted_class]}")

    return predicted_class, probabilities, sentiment_classes[predicted_class]


def main():
    """Main function to run the sentiment analysis model."""
    set_seed()
    model_path = "E:\\saved_models\\sentiment_analysis_transformer"

    model, tokenizer = load_model(model_path)
    if model is None:
        print("No existing model found. Training a new model...")

        df_combined = load_all_datasets()

        input_ids, attention_masks, labels, tokenizer = tokenize_data(df_combined)

        train_dataloader, validation_dataloader, test_dataloader = prepare_dataloaders(
            input_ids, attention_masks, labels
        )

        model = initialize_model()
        optimizer, scheduler = initialize_optimizer(model, train_dataloader)

        model, training_stats = train_model(
            model, train_dataloader, validation_dataloader, optimizer, scheduler
        )

        evaluate_model(model, test_dataloader)

        save_model(model, tokenizer, model_path)

    sample_text1 = "The stock market kept the same track for last 3 months."
    predict_sentiment(sample_text1, model, tokenizer)
    sample_text2 = "The stock market didn't keep its descending trend for last 3 months."
    predict_sentiment(sample_text2, model, tokenizer)
    sample_text3 = "A lot of people opened accounts at financial brokers in the last 3 months, which is a bad sign"
    predict_sentiment(sample_text3, model, tokenizer)



# Modified main function to incorporate negation handling
def main_with_negation_handling():
    """Modified main function with negation handling."""
    set_seed()
    model_path = "E:\\saved_models\\sentiment_analysis_transformer"

    model, tokenizer = load_model(model_path)
    if model is None:
        print("No existing model found. Training a new model...")

    predict_with_negation = apply_negation_handling(model, tokenizer, device)

    sample_text = "The stock market continued to fall for last 3 months."
    #sample_text = "The stock market didn't keep its descending trend for last 3 months."
    results = predict_with_negation(sample_text)

    print("\nOriginal model prediction:")
    print(f"Class: {results['original_class_name']} ({results['original_class']})")
    print(f"Probabilities: {results['original_probabilities']}")

    print("\nAdjusted prediction with negation handling:")
    print(f"Class: {results['adjusted_class_name']} ({results['adjusted_class']})")
    print(f"Probabilities: {results['adjusted_probabilities']}")
    print(f"Adjustment reason: {results['adjustment_reason']}")

    sample_text2 = "The company did not report losses this quarter."
    results2 = predict_with_negation(sample_text2)

    print("\nExample 2 - Original model prediction:")
    print(f"Class: {results2['original_class_name']} ({results2['original_class']})")
    print(f"Probabilities: {results2['original_probabilities']}")

    print("\nExample 2 - Adjusted prediction with negation handling:")
    print(f"Class: {results2['adjusted_class_name']} ({results2['adjusted_class']})")
    print(f"Probabilities: {results2['adjusted_probabilities']}")
    print(f"Adjustment reason: {results2['adjustment_reason']}")


if __name__ == "__main__":
    main_with_negation_handling()