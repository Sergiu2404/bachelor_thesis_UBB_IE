import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datetime import datetime
import time
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_financial_news():
    all_news = []

    sources = [
        {
            'name': 'Yahoo Finance',
            'url': 'https://finance.yahoo.com/news/',
            'article_selector': 'li.js-stream-content',
            'headline_selector': 'h3'
        },
        {
            'name': 'CNBC',
            'url': 'https://www.cnbc.com/finance/',
            'article_selector': '.Card-standardBreakerCard',
            'headline_selector': '.Card-title'
        },
        {
            'name': 'MarketWatch',
            'url': 'https://www.marketwatch.com/latest-news',
            'article_selector': '.article__content',
            'headline_selector': 'h3.article__headline'
        },
        {
            'name': 'Reuters Business',
            'url': 'https://www.reuters.com/business/',
            'article_selector': '.media-story-card',
            'headline_selector': '.media-story-card__heading__eqhp9'
        }
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Fetching financial news from websites...")

    for source in sources:
        try:
            response = requests.get(source['url'], headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.select(source['article_selector'])

                for article in articles:
                    headline_element = article.select_one(source['headline_selector'])
                    if headline_element:
                        headline = headline_element.get_text().strip()
                        if len(headline) > 10:
                            all_news.append(headline)

                print(f"Fetched {len(articles)} headlines from {source['name']}")
            else:
                print(f"Failed to fetch from {source['name']}")
        except Exception as e:
            print(f"Error fetching from {source['name']}: {str(e)}")

    predefined_news = [
        "Federal Reserve announces interest rate hike of 0.25%",
        "US GDP grows by 2.1% in the first quarter",
        "Tech stocks plummet as inflation concerns grow",
        "Oil prices reach 3-year high amid supply constraints",
        "Retail sales drop unexpectedly in April",
        "Housing market shows signs of cooling after record highs",
        "Corporate earnings exceed analyst expectations",
        "China imposes new regulations on tech giants",
        "Bitcoin falls below $30,000 for the first time in months",
        "European markets rally on stronger economic data",
        "Treasury yields hit new highs as investors flee bonds",
        "Unemployment rate falls to 3.8% in latest jobs report",
        "Inflation rate hits 5%, highest in over a decade",
        "Major bank announces 10,000 layoffs amid restructuring",
        "Supply chain issues continue to plague manufacturing sector",
        "Startup valuations drop 30% in private markets",
        "Consumer confidence index shows surprising resilience",
        "Gold prices surge as investors seek safe haven assets",
        "Airline stocks soar on increased travel demand",
        "Small business optimism reaches post-pandemic high",
        "Trump increases tariffs worldwide affecting global trade",
        "US dollar strengthens against basket of currencies",
        "Labor shortage pushes wages higher across industries",
        "Stock markets did not crash despite recession fears",
        "Specialists consider that stock market crash is close",
        "Economic recovery will take place in the next year",
        "Mortgage rates hit highest level in 15 years",
        "AI stocks surge on new technological breakthrough",
        "Energy sector faces pressure from renewable transition",
        "Central banks coordinate action to stabilize markets",
        "Consumer spending rises despite inflation pressures",
        "Manufacturing PMI indicates expansion in sector",
        "Semiconductor shortage expected to ease by year end",
        "Mergers and acquisitions hit record volume in quarter",
        "Crypto regulations tighten in major economies",
        "IPO market cools after record-breaking year",
        "Corporate bond yields spike on default concerns",
        "Venture capital funding slows in early-stage startups",
        "Government announces new infrastructure spending plan",
        "Agricultural commodities surge on weather concerns",
        "Retail bankruptcies increase as consumer habits shift",
        "Healthcare stocks rally on new drug approvals",
        "Commercial real estate vacancies hit decade high",
        "Initial jobless claims fall below pre-pandemic levels",
        "Foreign investment in US securities reaches new high",
        "Tesla stock jumps 15% after beating delivery estimates",
        "Dividend payouts reach record high among S&P 500 companies",
        "Geopolitical tensions disrupt global supply chains",
        "Federal budget deficit exceeds projections by 40%",
        "Trade deficit narrows as exports surge"
    ]

    all_news.extend(predefined_news)

    return list(set(all_news))


def generate_synthetic_news(template_list, n=1000):
    synthetic_news = []

    companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Tesla', 'Nvidia', 'JPMorgan',
                 'Bank of America', 'Walmart', 'ExxonMobil', 'Pfizer', 'Johnson & Johnson',
                 'Visa', 'Mastercard', 'IBM', 'Intel', 'AMD', 'Cisco', 'Oracle', 'Salesforce',
                 'Netflix', 'Disney', 'Coca-Cola', 'PepsiCo', 'McDonald\'s', 'Starbucks',
                 'Boeing', 'Lockheed Martin', 'General Motors', 'Ford', 'Toyota', 'Honda',
                 'Goldman Sachs', 'Morgan Stanley', 'Wells Fargo', 'Citigroup', 'UnitedHealth',
                 'CVS Health', 'Walgreens', 'Target', 'Home Depot', 'Lowe\'s']

    sectors = ['tech', 'finance', 'healthcare', 'retail', 'energy', 'consumer goods',
               'industrial', 'telecom', 'utilities', 'real estate', 'materials', 'automotive']

    positive_templates = [
        "{company} stock surges {percent}% after strong earnings report",
        "{company} exceeds analyst expectations with {percent}% revenue growth",
        "{company} announces expansion plans, shares up {percent}%",
        "{company} secures major deal worth billions",
        "{sector} sector rallies as {company} leads gains",
        "{company} introduces revolutionary new product to market acclaim",
        "Investors bullish on {company} after positive analyst coverage",
        "{company} raises dividend by {percent}%, signaling confidence",
        "{company} completes successful restructuring, boosting profit margins",
        "{company} stock hits all-time high on strong growth prospects",
        "Analysts upgrade {company} citing improved market conditions",
        "{company} beats quarterly revenue estimates by ${amount} million",
        "{sector} stocks surge led by {company}'s positive outlook",
        "{company} announces large share buyback program",
        "{company} wins regulatory approval for key product",
        "Market optimism grows as {company} reports record profits",
        "{company} successfully enters new market with strong initial sales",
        "{company} forms strategic partnership with {company2}",
        "Economic indicators point to strong growth in {sector} sector",
        "{company} reduces debt significantly, improves balance sheet"
    ]

    negative_templates = [
        "{company} stock plummets {percent}% after disappointing earnings",
        "{company} misses revenue targets, shares down {percent}%",
        "{company} announces layoffs affecting {amount} employees",
        "{company} faces regulatory investigation over business practices",
        "{sector} sector declines as {company} reports losses",
        "{company} recalls product due to safety concerns",
        "Investors bearish on {company} as competition intensifies",
        "{company} cuts dividend by {percent}%, raising concerns",
        "{company} restructuring fails to address fundamental issues",
        "{company} stock hits 52-week low amid market concerns",
        "Analysts downgrade {company} citing deteriorating conditions",
        "{company} misses quarterly estimates by ${amount} million",
        "{sector} stocks tumble following {company}'s negative guidance",
        "{company} halts share buyback program to preserve cash",
        "{company} denied regulatory approval for key product",
        "Market pessimism grows as {company} reports unexpected losses",
        "{company} struggles to enter new market with weak initial sales",
        "{company} terminates strategic partnership with {company2}",
        "Economic indicators point to contraction in {sector} sector",
        "{company} increases debt significantly, weakening balance sheet"
    ]

    neutral_templates = [
        "{company} reports earnings in line with expectations",
        "{company} maintains current workforce amid industry changes",
        "{company} awaits regulatory decision on proposed merger",
        "{sector} sector shows mixed results as {company} remains stable",
        "{company} refinances existing debt with similar terms",
        "{company} maintains market share despite competitive pressure",
        "Investors maintain hold rating on {company} stock",
        "{company} keeps dividend unchanged at quarterly meeting",
        "{company} implementing modest restructuring with minimal impact",
        "{company} stock trades within narrow range for third month",
        "Analysts maintain neutral stance on {company}",
        "{company} quarterly results mixed with some positives and negatives",
        "{sector} outlook uncertain according to {company} CEO",
        "{company} continues existing share repurchase program as planned",
        "{company} still in talks with regulators over product approval",
        "Market sentiment mixed regarding {company}'s future prospects",
        "{company} reports expected seasonal variation in sales figures",
        "{company} and {company2} continue existing partnership agreements",
        "Economic indicators suggest stable conditions in {sector} sector",
        "{company} debt levels remain consistent with previous quarter"
    ]

    if template_list == "positive":
        templates = positive_templates
    elif template_list == "negative":
        templates = negative_templates
    elif template_list == "neutral":
        templates = neutral_templates
    else:
        templates = positive_templates + negative_templates + neutral_templates

    for _ in range(n):
        template = random.choice(templates)
        company = random.choice(companies)
        company2 = random.choice([c for c in companies if c != company])
        sector = random.choice(sectors)
        percent = random.randint(3, 25)
        amount = random.randint(10, 500)

        news = template.format(company=company, company2=company2, sector=sector,
                               percent=percent, amount=amount)
        synthetic_news.append(news)

    return synthetic_news


def load_finbert_model():
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model


def analyze_sentiment(texts, tokenizer, model, batch_size=16):
    print("Analyzing sentiment of news articles...")
    results = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        for j, probs in enumerate(probabilities):
            sentiment_score = float(probs[0] - probs[1])

            sentiment_score = max(min(sentiment_score, 1.0), -1.0)
            sentiment_score = round(sentiment_score, 2)

            results.append({
                'text': batch_texts[j],
                'score': sentiment_score
            })

        time.sleep(0.1)

    return results


def balance_dataset(data, target_size=10000):
    print("Balancing dataset...")
    df = pd.DataFrame(data)

    bins = [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01]
    labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    df['sentiment_bin'] = pd.cut(df['score'], bins=bins, labels=labels)

    target_per_bin = target_size // len(labels)

    balanced_data = []

    for sentiment in labels:
        bin_data = df[df['sentiment_bin'] == sentiment]

        if len(bin_data) < target_per_bin:
            print(f"Generating synthetic {sentiment} news to reach target count")
            if sentiment in ['very_negative', 'negative']:
                synthetic_news = generate_synthetic_news("negative", target_per_bin - len(bin_data))
            elif sentiment in ['positive', 'very_positive']:
                synthetic_news = generate_synthetic_news("positive", target_per_bin - len(bin_data))
            else:
                synthetic_news = generate_synthetic_news("neutral", target_per_bin - len(bin_data))

            tokenizer, model = load_finbert_model()
            synthetic_results = analyze_sentiment(synthetic_news, tokenizer, model)

            synthetic_df = pd.DataFrame(synthetic_results)
            synthetic_df['sentiment_bin'] = pd.cut(synthetic_df['score'], bins=bins, labels=labels)
            filtered_synthetic = synthetic_df[synthetic_df['sentiment_bin'] == sentiment]

            needed = min(len(filtered_synthetic), target_per_bin - len(bin_data))
            balanced_data.extend(filtered_synthetic[['text', 'score']].head(needed).to_dict('records'))

            balanced_data.extend(bin_data[['text', 'score']].to_dict('records'))
        else:
            balanced_data.extend(bin_data[['text', 'score']].sample(target_per_bin).to_dict('records'))

    random.shuffle(balanced_data)

    return balanced_data[:target_size]


def create_financial_sentiment_dataset(target_size=10000):
    news_articles = get_financial_news()
    print(f"Collected {len(news_articles)} unique news articles")

    additional_synthetic = generate_synthetic_news("all", 5000)
    print(f"Generated {len(additional_synthetic)} synthetic news articles")

    all_news = news_articles + additional_synthetic
    all_news = list(set(all_news))
    print(f"Total of {len(all_news)} news articles for sentiment analysis")

    tokenizer, model = load_finbert_model()
    sentiment_results = analyze_sentiment(all_news, tokenizer, model)

    balanced_dataset = balance_dataset(sentiment_results, target_size)
    df = pd.DataFrame(balanced_dataset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"financial_news_sentiment_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print(f"Dataset created and saved as {filename}")
    print(f"Final dataset size: {len(df)} records")

    print("\nSentiment Score Distribution:")
    print(df['score'].describe())

    print("\nSample of the dataset:")
    print(df.sample(10))

    return df


# if __name__ == "__main__":
#     create_financial_sentiment_dataset(10000)







#
# import pandas as pd
# from datasets import load_dataset
# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# import torch
# import os
#
# # Load FinBERT model
# model_name = "yiyanghkust/finbert-tone"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
#
# # Set up FinBERT pipeline
# device = 0 if torch.cuda.is_available() else -1
# finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
#
# # Function to get sentiment score between [-1, 1]
# def get_continuous_score(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()
#
#     # Index: 0=negative, 1=neutral, 2=positive
#     score = (probs[2] - probs[0])  # from [-1,1]
#     return round(score, 2)
#
# # 1. Load your existing financial news dataset (leave scores as-is)
# custom_path = "sentiment_datasets/financial_news_sentiment.csv"
# custom_df = pd.read_csv(custom_path)
#
# # 2. Load and process Twitter dataset
# twitter_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
# twitter_df = pd.DataFrame(twitter_dataset)[['text']]
#
# # Apply FinBERT to Twitter data (you can reduce rows for speed)
# twitter_df['score'] = twitter_df['text'].apply(get_continuous_score)
#
# # 3. Merge datasets (keep only 'text' and 'score' columns)
# custom_df = custom_df[['text', 'score']]
# twitter_df = twitter_df[['text', 'score']]
#
# all_df = pd.concat([custom_df, twitter_df], ignore_index=True).drop_duplicates(subset=["text"])
#
# # 4. Save final merged dataset
# os.makedirs("sentiment_datasets", exist_ok=True)
# final_path = "sentiment_datasets/all_financial_news_sentiment_datasets.csv"
# all_df.to_csv(final_path, index=False)
#
# print(f"✅ Merged dataset saved to {final_path}")





#
# import os
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from datasets import load_dataset
# import logging
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# os.makedirs('sentiment_datasets', exist_ok=True)
#
#
# def score_text_with_finbert(texts, model, tokenizer):
#     logger.info(f"Scoring {len(texts)} texts with FinBERT")
#     results = []
#
#     batch_size = 32
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]
#         batch_texts = [str(text) for text in batch_texts]
#
#         encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
#
#         if torch.cuda.is_available():
#             encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
#             model = model.cuda()
#
#         with torch.no_grad():
#             outputs = model(**encoded_input)
#
#         probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
#
#         for prob in probs:
#             score = -1 * prob[0] + 0 * prob[1] + 1 * prob[2]
#             results.append(float(score))
#
#         if (i // batch_size) % 5 == 0:
#             logger.info(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
#
#     return results
#
#
# try:
#     logger.info("Loading existing financial news sentiment data")
#     existing_df = pd.read_csv('sentiment_datasets/financial_news_sentiment.csv', names=['text', 'score'])
#     logger.info(f"Loaded {len(existing_df)} existing entries")
# except FileNotFoundError:
#     logger.warning("Existing sentiment file not found. Creating a new one.")
#     existing_df = pd.DataFrame(columns=['text', 'score'])
#
#     sample_data = [
#         ["AMD reduces debt significantly, improves balance sheet", 0.93],
#         ["Economic indicators point to contraction in telecom sector", -0.96],
#         ["telecom sector rallies as Tesla leads gains", 0.86],
#         ["Investors maintain hold rating on Google stock", 0.01],
#         ["Meta restructuring fails to address fundamental issues", -0.3],
#         ["CVS Health introduces revolutionary new product to market acclaim", 0.58],
#         ["Target beats quarterly revenue estimates by $92 million", 0.55],
#         ["real estate sector rallies as Honda leads gains", 0.5],
#         ["Wells Fargo introduces revolutionary new product to market acclaim", 0.62],
#         ["Investors bearish on Google as competition intensifies", -0.92],
#         ["Cisco quarterly results mixed with some positives and negatives", -0.94]
#     ]
#     existing_df = pd.DataFrame(sample_data, columns=['text', 'score'])
#     existing_df.to_csv('sentiment_datasets/financial_news_sentiment.csv', index=False, header=False)
#     logger.info(f"Created sample file with {len(existing_df)} entries")
#
# logger.info("Loading FinBERT model")
# finbert_model_name = "ProsusAI/finbert"
# tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
# model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
#
# dfs_to_merge = [existing_df]
#
# financial_datasets = [
#     "financial_phrasebank",
#     "zeroshot/twitter-financial-news-sentiment"
# ]
#
# for dataset_name in financial_datasets:
#     logger.info(f"Attempting to load {dataset_name} dataset")
#     try:
#         if dataset_name == "financial_phrasebank":
#             dataset = load_dataset(dataset_name, name="sentences_allagree")
#             if 'train' in dataset:
#                 df = pd.DataFrame({
#                     'text': dataset['train']['sentence'],
#                     'original_label': dataset['train']['label']
#                 })
#                 logger.info(f"Loaded Financial PhraseBank with {len(df)} entries")
#             else:
#                 logger.warning(f"No 'train' split found in {dataset_name}")
#                 continue
#
#         elif "twitter-financial-news-sentiment" in dataset_name:
#             dataset = load_dataset(dataset_name)
#             if 'train' in dataset:
#                 df = pd.DataFrame({
#                     'text': dataset['train']['text'],
#                     'original_label': dataset['train']['label'] if 'label' in dataset['train'].features else
#                     dataset['train']['sentiment'] if 'sentiment' in dataset['train'].features else [0] * len(
#                         dataset['train'])
#                 })
#                 logger.info(f"Loaded Twitter Financial News with {len(df)} entries")
#             else:
#                 logger.warning(f"No 'train' split found in {dataset_name}")
#                 continue
#
#         df['score'] = score_text_with_finbert(df['text'].tolist(), model, tokenizer)
#
#         df = df[['text', 'score']]
#
#         logger.info(f"Processed {len(df)} entries from {dataset_name}")
#         dfs_to_merge.append(df)
#     except Exception as e:
#         logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
#
# logger.info("Loading FiQA sentiment classification dataset")
# try:
#     fiqa_dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
#
#     fiqa_dfs = []
#     for split in ['train', 'test', 'validation']:
#         if split in fiqa_dataset:
#             columns = fiqa_dataset[split].column_names
#             logger.info(f"Available columns in FiQA {split} split: {columns}")
#
#             if 'sentence' in columns and 'score' in columns:
#                 split_df = pd.DataFrame({
#                     'text': fiqa_dataset[split]['sentence'],
#                     'original_score': fiqa_dataset[split]['score']
#                 })
#                 min_score = split_df['original_score'].min()
#                 max_score = split_df['original_score'].max()
#
#                 if min_score < -1 or max_score > 1:
#                     logger.info(f"Normalizing FiQA scores from range [{min_score}, {max_score}] to [-1, 1]")
#                     split_df['score'] = split_df['original_score'].apply(
#                         lambda x: 2 * (x - min_score) / (max_score - min_score) - 1 if max_score != min_score else 0
#                     )
#                 else:
#                     split_df['score'] = split_df['original_score']
#             else:
#                 split_df = pd.DataFrame({
#                     'text': fiqa_dataset[split]['sentence']
#                 })
#                 split_df['score'] = score_text_with_finbert(split_df['text'].tolist(), model, tokenizer)
#
#             split_df = split_df[['text', 'score']]
#             fiqa_dfs.append(split_df)
#
#     if fiqa_dfs:
#         fiqa_df = pd.concat(fiqa_dfs, ignore_index=True)
#         logger.info(f"Processed {len(fiqa_df)} entries from FiQA dataset")
#         dfs_to_merge.append(fiqa_df)
#     else:
#         logger.warning("No usable data found in FiQA dataset")
#
# except Exception as e:
#     logger.error(f"Error loading FiQA dataset: {str(e)}")
#
# logger.info("Merging all datasets")
# if dfs_to_merge:
#     for i, df in enumerate(dfs_to_merge):
#         if 'text' not in df.columns or 'score' not in df.columns:
#             logger.warning(f"Skipping dataframe at index {i} due to missing required columns. Columns: {df.columns}")
#             dfs_to_merge[i] = None
#
#     dfs_to_merge = [df for df in dfs_to_merge if df is not None]
#
#     if dfs_to_merge:
#         final_df = pd.concat(dfs_to_merge, ignore_index=True)
#
#         before_dedup = len(final_df)
#         final_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
#         after_dedup = len(final_df)
#         logger.info(f"Removed {before_dedup - after_dedup} duplicate entries")
#
#         final_df['score'] = pd.to_numeric(final_df['score'], errors='coerce')
#         final_df = final_df.dropna(subset=['score'])  # Drop rows with non-numeric scores
#
#         final_df['score'] = final_df['score'].apply(lambda x: max(min(float(x), 1.0), -1.0))
#
#         final_df['score'] = final_df['score'].round(2)
#
#         output_path = 'sentiment_datasets/all_financial_news_sentiment_datasets.csv'
#         final_df.to_csv(output_path, index=False, header=False)
#
#         logger.info(f"Successfully created merged dataset with {len(final_df)} entries at {output_path}")
#
#         print("\nSample of final dataset:")
#         print(final_df.sample(min(10, len(final_df))).to_string())
#     else:
#         logger.error("No valid dataframes to merge")
# else:
#     logger.error("No dataframes to merge")





import pandas as pd
import random

input_path = "./sentiment_datasets/all_financial_sentiment_datasets_negation_handling.csv"
output_path = "./sentiment_datasets/all_financial_sentiment_datasets_major_handling.csv"

df_original = pd.read_csv(input_path, header=None, names=["text", "sentiment"])

augmented_data = [
    ("TTrump administration will cause a major stock market crash due to their tariffs for all the other countries.", -0.7),
    ("The current US republican administration could cause a major stock market crash due to their tariffs imposed for all the other countries.", -0.7),
    ("Germany will spend more money on defend industry, weapons and army", -0.2),
    ("The European Union member countries will spend more money on defend industry, weapons and army", 0.2),
    ("US stock market could crash anytime soon", -0.8),
    ("Chinese stock market could crash anytime soon", -0.8),
    ("The global economy shows real signs of recovery", 0.7),
    ("Investors see great chances of a steep fall in stock prices", -0.9),
    ("The possibility of a significant downturn is growing", -0.8),
    ("Dividends are expected to decrease", -0.8),
    ("Dividends are not expected to increase", -0.2),
    ("There is no clear sign of misinformation spreading", 0.3),
    ("Misinformation about market trends is concerning investors", -0.4),
    ("Efforts to fight misinformation are making a difference", 0.4),
    ("It's unlikely that investors won't gain this quarter", 0.9),
    ("Analysts believe the market won't recover", -0.8),
    ("Investors don't think dividends will not be decreased this year", -0.5),
    ("Investors want to minimize their losses and will quit the market", -0.9),
    ("Investors want to mark their profits and will quit the market until the situation recovers", -0.8),
]

expanded_data = []
for _ in range(10):
    for text, score in augmented_data:
        variation = text.replace("stock market", random.choice(["financial markets", "global markets", "investment sector"]))
        variation = variation.replace("dividends", random.choice(["returns", "dividends", "yields"]))
        variation = variation.replace("investors", random.choice(["investors", "analysts", "traders", "economists"]))
        variation = variation.replace("could", random.choice(["might", "could", "may"]))
        noise = random.uniform(-0.05, 0.05)
        new_score = min(max(score + noise, -1), 1)
        expanded_data.append((variation, round(new_score, 3)))

df_augmented = pd.DataFrame(expanded_data, columns=["text", "sentiment"])

df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
df_combined.to_csv(output_path, index=False, header=False)