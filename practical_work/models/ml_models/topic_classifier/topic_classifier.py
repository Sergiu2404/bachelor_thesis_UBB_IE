from rapidfuzz import process
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def load_ticker_db(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['Symbol', 'Name']].dropna()
    return df

def extract_company_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "ORG"]


def match_company_to_ticker(company_name, ticker_df, threshold=80):
    choices = ticker_df['Name'].tolist()
    match, score, idx = process.extractOne(company_name, choices)
    if score >= threshold:
        return ticker_df.iloc[idx]['Symbol'], match
    return None, None


def classify_and_get_ticker(text, ticker_df):
    company_names = extract_company_names(text)
    results = []

    for name in company_names:
        ticker, matched_name = match_company_to_ticker(name, ticker_df)
        if ticker:
            results.append((name, matched_name, ticker))

    return results

ticker_df = load_ticker_db("./nasdaq_screener_1745958373961.csv")

text = "Apple Inc. announced strong earnings while Alcoa is expanding production."

companies = classify_and_get_ticker(text, ticker_df)

for original, matched, ticker in companies:
    print(f"Detected: {original} → Matched: {matched} → Ticker: {ticker}")
