import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import json
import time
import random


def load_fiqa_dataset():
    print("Loading FiQA dataset...")
    splits = {
        'train': 'data/train-00000-of-00001-aeefa1eadf5be10b.parquet',
        'test': 'data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet',
        'valid': 'data/valid-00000-of-00001-51867fe1ac59af78.parquet'
    }

    df_fiqa = pd.read_parquet("hf://datasets/TheFinAI/fiqa-sentiment-classification/" + splits["train"])

    df_fiqa = df_fiqa[['sentence', 'score']].rename(columns={'sentence': 'text', 'score': 'sentiment'})

    return df_fiqa


def setup_zero_shot_classifier():
    print("Setting up zero-shot classifier...")

    model_name = "facebook/bart-large-mnli"

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    return classifier


def generate_longevity_dataset_with_zero_shot(df, classifier, sample_size=None, batch_size=16):
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42)

    texts = df['text'].tolist()
    print(f"Processing {len(texts)} texts with zero-shot classification...")

    candidate_labels = [
        "very short-term financial impact (days to weeks)",
        "short-term financial impact (weeks to months)",
        "medium-term financial impact (months to a year)",
        "long-term financial impact (1-3 years)",
        "very long-term financial impact (3+ years)"
    ]

    all_scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]
        batch_results = []

        for text in batch:
            try:
                result = classifier(text, candidate_labels)

                label_values = {
                    "very short-term financial impact (days to weeks)": 0.1,
                    "short-term financial impact (weeks to months)": 0.3,
                    "medium-term financial impact (months to a year)": 0.5,
                    "long-term financial impact (1-3 years)": 0.7,
                    "very long-term financial impact (3+ years)": 0.9
                }

                weighted_score = sum(score * label_values[label]
                                     for label, score in zip(result['labels'], result['scores']))

                batch_results.append(weighted_score)

            except Exception as e:
                print(f"Error processing text: {e}")
                batch_results.append(random.uniform(0.4, 0.6))

        all_scores.extend(batch_results)

    longevity_df = pd.DataFrame({
        'text': texts,
        'longevity': all_scores
    })

    return longevity_df


def fine_tune_bert_for_longevity(sample_size=100):
    print("\nCreating synthetic dataset for fine-tuning...")

    synthetic_data = [
        {"text": "The stock just dropped 5% in after-hours trading.", "longevity": 0.1},
        {"text": "Today's market volatility has traders on edge.", "longevity": 0.1},
        {"text": "The company missed quarterly earnings by 2 cents.", "longevity": 0.1},
        {"text": "Trading was halted briefly due to circuit breakers.", "longevity": 0.1},
        {"text": "Day traders are piling into this momentum stock.", "longevity": 0.1},

        {"text": "The upcoming earnings report will likely impact Q2 performance.", "longevity": 0.3},
        {"text": "Analysts expect a dividend increase next quarter.", "longevity": 0.3},
        {"text": "The company is facing supply chain issues that should resolve within months.", "longevity": 0.3},
        {"text": "Quarterly revenue is projected to increase by 15%.", "longevity": 0.3},
        {"text": "The recent product recall will affect this quarter's margins.", "longevity": 0.3},

        {"text": "The new CEO outlined a 12-month plan to cut costs.", "longevity": 0.5},
        {"text": "Annual projections suggest moderate growth through next fiscal year.", "longevity": 0.5},
        {"text": "The company is expanding into new markets by year-end.", "longevity": 0.5},
        {"text": "This year's budget allocates more to R&D than previous years.", "longevity": 0.5},
        {"text": "The annual restructuring plan should improve margins within a year.", "longevity": 0.5},

        {"text": "The 3-year strategic plan focuses on sustainable growth.", "longevity": 0.7},
        {"text": "Capital investments will build manufacturing capacity for years to come.", "longevity": 0.7},
        {"text": "The company is developing new technology with a 2-year timeline to market.", "longevity": 0.7},
        {"text": "Long-term debt has been restructured with favorable 5-year terms.", "longevity": 0.7},
        {"text": "The new factory will increase production capacity for the next 3 years.", "longevity": 0.7},

        {"text": "The company's 10-year vision focuses on becoming carbon neutral.", "longevity": 0.9},
        {"text": "This acquisition is part of a multi-decade growth strategy.", "longevity": 0.9},
        {"text": "Investments in R&D will position the company for the next generation.", "longevity": 0.9},
        {"text": "The 20-year infrastructure plan will transform the company's capabilities.", "longevity": 0.9},
        {"text": "These patents will protect the company's market position for years to come.", "longevity": 0.9},
    ]

    for _ in range(75):
        category = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        if category == 0.1:
            timeframes = ["today", "this morning", "yesterday", "right now", "this hour"]
            actions = ["jumped", "dropped", "surged", "plummeted", "rallied"]
            subjects = ["stock price", "trading volume", "market sentiment", "investor interest"]
            text = f"The {random.choice(subjects)} {random.choice(actions)} {random.choice(timeframes)}."
        elif category == 0.3:
            timeframes = ["this quarter", "next month", "in the coming weeks", "by quarter end"]
            actions = ["expects", "projects", "targets", "anticipates"]
            metrics = ["earnings", "revenue", "profit margins", "sales figures", "quarterly results"]
            text = f"The company {random.choice(actions)} {random.choice(metrics)} to improve {random.choice(timeframes)}."
        elif category == 0.5:
            timeframes = ["this year", "in the next 12 months", "by fiscal year end", "within a year"]
            strategy = ["annual plan", "yearly targets", "fiscal strategy", "yearly roadmap"]
            goals = ["market expansion", "cost cutting", "revenue growth", "profit improvement"]
            text = f"The {random.choice(strategy)} focuses on {random.choice(goals)} {random.choice(timeframes)}."
        elif category == 0.7:
            timeframes = ["next 2-3 years", "multi-year plan", "by 2027", "over the next 36 months"]
            strategy = ["medium-term strategy", "3-year roadmap", "strategic direction"]
            initiatives = ["digital transformation", "international expansion", "product diversification"]
            text = f"The {random.choice(strategy)} includes {random.choice(initiatives)} over the {random.choice(timeframes)}."
        else:  # 0.9
            timeframes = ["next decade", "coming generation", "long-term future", "next 10+ years"]
            vision = ["long-term vision", "multi-decade plan", "generational strategy"]
            goals = ["sustainability targets", "market leadership", "industry transformation"]
            text = f"The company's {random.choice(vision)} aims at {random.choice(goals)} for the {random.choice(timeframes)}."

        synthetic_data.append({"text": text, "longevity": category})

    synthetic_dataset = Dataset.from_pandas(pd.DataFrame(synthetic_data))

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = synthetic_dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("longevity", "labels")

    training_args = TrainingArguments(
        output_dir="E:\saved_models\longevity_model",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    print("\nFine-tuning BERT for longevity prediction...")
    trainer.train()

    model_path = "E:\saved_models\longevity_model"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Saved fine-tuned model to {model_path}")

    return model, tokenizer


def predict_longevity_with_bert(texts, model, tokenizer, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting longevity scores"):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()

            predictions = np.clip(predictions, 0, 1)

        all_scores.extend(predictions.tolist())

    return all_scores


def create_longevity_dataset(use_fine_tuned=True, sample_size=None):
    os.makedirs("./sentiment_datasets", exist_ok=True)

    df_fiqa = load_fiqa_dataset()

    if use_fine_tuned:
        model, tokenizer = fine_tune_bert_for_longevity()

        if sample_size is not None:
            df_fiqa = df_fiqa.sample(sample_size, random_state=42)

        texts = df_fiqa['text'].tolist()
        longevity_scores = predict_longevity_with_bert(texts, model, tokenizer)

        longevity_df = pd.DataFrame({
            'text': texts,
            'longevity': longevity_scores
        })
    else:
        classifier = setup_zero_shot_classifier()
        longevity_df = generate_longevity_dataset_with_zero_shot(df_fiqa, classifier, sample_size)

    output_path = "./sentiment_datasets/longevity_dataset.csv"
    longevity_df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

    longevity_df.to_parquet("./sentiment_datasets/longevity_dataset.parquet", index=False)
    print(f"Dataset also saved as parquet to ./sentiment_datasets/longevity_dataset.parquet")

    return longevity_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate a text longevity dataset from FiQA')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to process (for testing)')
    parser.add_argument('--method', type=str, choices=['fine-tuned', 'zero-shot'], default='fine-tuned',
                        help='Method to use for longevity classification')
    args = parser.parse_args()

    print(f"Using method: {args.method}")
    if args.sample:
        print(f"Processing sample of {args.sample} texts")

    use_fine_tuned = (args.method == 'fine-tuned')
    df = create_longevity_dataset(use_fine_tuned=use_fine_tuned, sample_size=args.sample)

    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Longevity score distribution:")
    print(f"  Min: {df['longevity'].min():.2f}")
    print(f"  Max: {df['longevity'].max():.2f}")
    print(f"  Mean: {df['longevity'].mean():.2f}")
    print(f"  Median: {df['longevity'].median():.2f}")

    bins = [0, 0.25, 0.5, 0.75, 1.0]
    labels = ['Very short-term', 'Short-term', 'Long-term', 'Very long-term']
    df['category'] = pd.cut(df['longevity'], bins=bins, labels=labels)
    distribution = df['category'].value_counts(normalize=True) * 100

    print("\nDistribution by category:")
    for category, percentage in distribution.items():
        print(f"  {category}: {percentage:.1f}%")