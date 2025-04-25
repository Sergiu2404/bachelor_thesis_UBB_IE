import random
import pandas as pd
import numpy as np

# Word Lists
positive_words = [
    "increase", "grow", "progress", "improve", "boost", "advance", "expand", "rise", "gain", "surge", 'beat', 'boost',
    'exceed', 'surprisingly', 'grow', 'up', 'rise', 'gain', 'profitable', 'hike', 'well',
    'earn', 'strong', 'strength', 'higher', 'high', 'rally', 'bullish', 'outperform', 'surpass', 'good',
    'opportunity', 'success', 'improve', 'breakthrough', 'progress', 'upgrade', 'increase', 'win', 'reward',
    'advance', 'progress', 'soar', 'climb', 'ascent', 'great', 'amazing', 'above', 'expansion', 'expand',
    'optimistic', 'wide', 'jump', 'double', 'triple', 'twice', 'upside', 'outperform',
    # Additional positive words
    'recover', 'rebound', 'thrive', 'prosper', 'flourish', 'excel', 'robust', 'favorable', 'promising',
    'peak', 'superior', 'outstanding', 'impressive', 'maximize', 'accelerate', 'momentum', 'efficient',
    'premium', 'advantage', 'innovative', 'leadership', 'breakthrough', 'resilient', 'stability'
]

negative_words = [
    "decrease", "drop", "decline", "fall", "plunge", "reduce", "worsen", "deteriorate", "contract", "slump",
    'miss', 'disappoint', 'decline', 'decrease', 'loss', 'negative', 'descent', 'low', 'slow', 'down', 'slowdown',
    'weak', 'drop', 'fall', 'bearish', 'underperform', 'risk', 'dive', 'problem', 'below', 'downside', 'headwind',
    'warning', 'fail', 'bankruptcy', 'investigation', 'lawsuit', 'litigation', 'pressure', 'fear', 'recession', 'bad',
    'concern', 'caution', 'downgrade', 'decrease', 'lose', 'challenge', 'bearish', 'degeneration', 'half', 'downside',
    # Additional negative words
    'collapse', 'crash', 'debt', 'deficit', 'default', 'layoff', 'liquidation', 'obsolete', 'penalty',
    'poor', 'shortage', 'shrink', 'strain', 'struggle', 'suspend', 'taper', 'tension', 'terminate',
    'threat', 'uncertain', 'underperform', 'unstable', 'vulnerability', 'volatile', 'writedown'
]

neutral_words = [
    "report", "statement", "data", "meeting", "announcement", "update", "summary", "event", "discussion", "figure",
    'unchanged', 'neutral', 'maintain', 'expect', 'keep',
    'estimate', 'guidance', 'target', 'announce', 'report', 'quarter',
    'fiscal', 'year', 'slightly', 'slight', 'pretty', 'slow', 'prudent', 'insignificant', 'unsignificant',
    # Additional neutral words
    'forecast', 'projection', 'outlook', 'assessment', 'evaluation', 'analysis', 'consideration',
    'review', 'schedule', 'standard', 'traditional', 'typical', 'usual', 'common', 'average',
    'median', 'moderate', 'ordinary', 'regular', 'steady', 'routine', 'conventional'
]

positive_phrases = [
    "dividends increase", "market growth", "economic expansion", "investment rise",
    "stock surge", "profits improve", "earnings boost", "business growth",
    "company's profits grow", "strong market performance", "dividends grow", "economy improves",
    "earnings rise", "company outperforms expectations", "strong earnings growth",
    "expansion of business", "investment value up", "economic boom", "stock price increases",
    "assets increase", "investor confidence rises", "positive financial reports",
    "boost in profits", "strong economic indicators", "company performance exceeds",
    "successful market strategies", "fiscal strength", "substantial growth",
    "strong future outlook", "strong bullish sentiment", "market rally", "upward trend",
    "optimistic market forecast", "profit margins rise", "revenue increases",
    "business revenue grows", "investor optimism increases", "positive economic outlook",
    # Additional positive phrases
    "decrease losses", "reduce risk", "cut expenses", "lower costs", "diminish debt",
    "shrink deficit", "minimize exposure", "decline in expenses", "falling costs",
    "efficient cost structure", "strategic restructuring", "favorable refinancing",
    "streamlined operations", "cost optimization", "enhanced productivity"
]

negative_phrases = [
    "market crash", "earnings decline", "economic slowdown", "investment drop",
    "stock plunge", "revenue loss", "financial deterioration", "credit risk",
    "company's profits fall", "earnings dip", "poor financial performance",
    "economic downturn", "stock price drops", "business decline", "rising risks",
    "negative economic forecast", "revenue reduction", "business performance worsens",
    "declining stock value", "weak business results", "market contraction", "fall in market sentiment",
    "market collapse", "decreasing dividends", "loss of investor confidence",
    "negative quarterly results", "economic regression", "bear market", "downward trend",
    "negative financial forecast", "declining revenue", "company performance drops",
    "financial crisis", "bankruptcy concerns", "unexpected financial losses",
    "business suffering losses", "financial stress increases", "declining profitability",
    "stock price plummets", "economic contraction", "revenue disappointment",
    "investor fear increases", "recession risk grows", "market uncertainties",
    "market hesitation", "weak economic indicators", "economic recession",
    # Additional negative phrases
    "increase deficit", "rise in debt", "growth in liabilities", "higher debt burden",
    "expanding costs", "boost in expenses", "climbing unemployment", "surge in inflation",
    "improved bankruptcy numbers", "accelerating downturn", "robust sell-off",
    "widening trade deficit", "strengthening headwinds", "progressive losses"
]

neutral_phrases = [
    "financial report", "economic data", "market update", "trading volume",
    "policy statement", "federal meeting", "quarterly results", "company report",
    "financial disclosure", "company earnings report", "market overview",
    "market analysis report", "year-end review", "quarterly earnings call",
    "economic statement", "company performance update", "reporting season",
    "economic indicators", "business strategy", "financial outlook",
    "financial forecast", "market performance analysis", "investment report",
    "company guidance", "fiscal report", "target estimate", "economic performance",
    "market trends", "financial outlook forecast", "corporate earnings",
    "economic policy update", "government report", "market expectation",
    "investment outlook", "annual report", "market research update",
    "company earnings forecast", "strategic investment planning", "market condition assessment",
    "economic forecast", "investment activity", "sector report",
    "company's fiscal performance", "business environment report", "capital markets analysis",
    # Additional neutral phrases
    "analyst consensus", "market assessment", "industry benchmark", "comparative analysis",
    "financial metrics", "balance sheet items", "liquidity measures", "valuation model",
    "technical indicators", "market capitalization", "shareholder structure", "regulatory filing"
]

# Tricky phrases with their sentiment values
tricky_phrases = {
    # Positive words with negative context (negative sentiment)
    "increase deficit": -0.7, "rise in debt": -0.6, "increase in bankruptcy": -0.8,
    "increase in losses": -0.7, "surge in debts": -0.7, "boost in negative returns": -0.6,
    "growth in liabilities": -0.6, "rising debt levels": -0.7, "expanding trade deficit": -0.5,
    "improving bankruptcy numbers": -0.6, "strong decline": -0.8, "accelerating downturn": -0.7,
    "robust sell-off": -0.6, "higher unemployment": -0.7, "growing concerns": -0.6,
    "surge in inflation": -0.7, "boost to tax burden": -0.5, "increased volatility": -0.6,
    "positive cash burn rate": -0.5, "economic boom with high unemployment": -0.3,
    "growth with rising inflation": -0.4,

    # Negative words with positive context (positive sentiment)
    "decrease losses": 0.6, "improve decline": 0.5, "reduce risk": 0.7,
    "decrease in expenses": 0.7, "falling interest rates": 0.6, "declining unemployment": 0.8,
    "reduced tax burden": 0.7, "negative interest rates boosting economy": 0.5,
    "shrinking deficit": 0.6, "lower inflation": 0.7, "slowing debt accumulation": 0.5,
    "bearish signals ended": 0.6, "downward adjustment of overvalued assets": 0.4,
    "cutting excessive spending": 0.6, "slumping prices benefit consumers": 0.5,
    "halting unprofitable ventures": 0.6, "collapse in production costs": 0.7,
    "downgrade reversed": 0.6
}


def pos_score():
    return round(random.uniform(0.5, 0.9), 2)


def neg_score():
    return round(random.uniform(-0.9, -0.5), 2)


def neu_score():
    return round(random.uniform(-0.1, 0.1), 2)


def flip_score(score):
    return round(-score + random.uniform(-0.1, 0.1), 2)


negations = ["not", "no", "never", "neither", "n't", "hardly", "scarcely", "barely"]


def create_enhanced_financial_dataset():
    dataset = []

    # Add simple words with their scores
    for word in positive_words:
        dataset.append((word, pos_score()))
    for word in negative_words:
        dataset.append((word, neg_score()))
    for word in neutral_words:
        dataset.append((word, neu_score()))

    # Add phrases with their scores
    for phrase in positive_phrases:
        dataset.append((phrase, pos_score()))
    for phrase in negative_phrases:
        dataset.append((phrase, neg_score()))
    for phrase in neutral_phrases:
        dataset.append((phrase, neu_score()))

    # Add tricky phrases with their predefined scores
    for phrase, score in tricky_phrases.items():
        dataset.append((phrase, score))

    # Create single negations
    all_items = dataset.copy()
    for text, score in all_items:
        for neg in negations:
            if "n't" in neg:
                words = text.split()
                if len(words) > 1:
                    negated = f"{words[0]}{neg} {' '.join(words[1:])}"
                else:
                    negated = f"{text}{neg}"
            elif neg in ["neither", "nor"]:
                negated = f"{neg} {text}"
            else:
                negated = f"{neg} {text}"
            dataset.append((negated, flip_score(score)))

    # Create double negations (bringing sentiment back close to original)
    all_items = dataset.copy()
    for text, score in all_items:
        # Only create double negations for items that don't already have negations
        if not any(neg in text.lower() for neg in negations):
            # Apply two different negations
            for i, neg1 in enumerate(negations):
                for neg2 in negations[i + 1:]:  # Use different negation patterns
                    if neg1 == "n't" or neg2 == "n't":
                        continue  # Skip complex contractions for double negations

                    # Create double negated text - various patterns
                    if random.random() < 0.5:
                        # Pattern: "not never X"
                        double_negated = f"{neg1} {neg2} {text}"
                    else:
                        # Pattern: "not X that isn't Y"
                        words = text.split()
                        if len(words) > 1:
                            middle = len(words) // 2
                            double_negated = f"{neg1} {' '.join(words[:middle])} that {neg2} {' '.join(words[middle:])}"
                        else:
                            double_negated = f"{neg1} {text} that is {neg2} true"

                    # Double negation mostly returns to original sentiment but with some variation
                    # For positive/negative terms, come back to slightly less extreme
                    if score > 0.3:  # Was positive
                        new_score = score * 0.8 + random.uniform(-0.1, 0.1)
                    elif score < -0.3:  # Was negative
                        new_score = score * 0.8 + random.uniform(-0.1, 0.1)
                    else:  # Was neutral
                        new_score = score + random.uniform(-0.1, 0.1)

                    dataset.append((double_negated, round(new_score, 2)))

    # Create financial news templates with these expressions
    templates = [
        "Company reports {} in quarterly results",
        "Analysts note {} in latest economic data",
        "CEO discusses {} during earnings call",
        "Market reacts to {} announcement",
        "Investors concerned about {}",
        "Report highlights {} trend in sector",
        "Financial Times: {} becoming more common",
        "Wall Street Journal reports {} in major companies",
        "The economy shows signs of {}, experts say",
        "New policy could lead to {}, according to analysis"
    ]

    news_dataset = []
    sample_items = random.sample(dataset, min(len(dataset), 5000))  # Limit to avoid huge dataset

    for text, score in sample_items:
        template = random.choice(templates)
        news_item = template.format(text)
        news_dataset.append((news_item, score))

    final_dataset = dataset + news_dataset
    random.shuffle(final_dataset)

    df = pd.DataFrame(final_dataset, columns=['text', 'sentiment'])
    return df


if __name__ == "__main__":
    df = create_enhanced_financial_dataset()
    print(f"Created dataset with {len(df)} entries")
    print(df.head(10))

    # Sample distribution statistics
    pos_count = len(df[df['sentiment'] > 0.3])
    neg_count = len(df[df['sentiment'] < -0.3])
    neu_count = len(df[(df['sentiment'] >= -0.3) & (df['sentiment'] <= 0.3)])

    print(f"\nPositive examples: {pos_count} ({pos_count / len(df) * 100:.1f}%)")
    print(f"Negative examples: {neg_count} ({neg_count / len(df) * 100:.1f}%)")
    print(f"Neutral examples: {neu_count} ({neu_count / len(df) * 100:.1f}%)")

    df.to_csv("enhanced_financial_sentiment_dataset.csv", index=False)
    print("\nDataset saved to enhanced_financial_sentiment_dataset.csv")

    double_neg_examples = df[df['text'].str.contains('not.*never|never.*not|not.*no |no .*not')]
    if len(double_neg_examples) > 0:
        print("\nSample double negation examples:")
        for i, (text, score) in enumerate(double_neg_examples.head(5).values):
            print(f"{i + 1}. Text: '{text}' - Score: {score}")


