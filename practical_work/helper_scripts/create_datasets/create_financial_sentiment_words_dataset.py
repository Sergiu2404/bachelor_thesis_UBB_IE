import csv
import random
import os

import random
import pandas as pd
import numpy as np

positive_words = [
    "increase", "grow", "progress", "improve", "boost", "advance", "expand", "rise", "gain", "surge", 'beat', 'boost', 'exceed', 'surprisingly', 'grow', 'up', 'rise', 'gain', 'profitable', 'hike', 'well',
    'earn', 'strong', 'strength', 'higher', 'high', 'rally', 'bullish', 'outperform', 'surpass', 'good',
    'opportunity', 'success', 'improve', 'breakthrough', 'progress', 'upgrade', 'increase', 'win', 'reward',
    'advance', 'progress', 'soar', 'climb', 'ascent', 'great', 'amazing', 'above', 'expansion', 'expand',
    'optimistic', 'wide', 'jump', 'double', 'triple', 'twice', 'upside', 'outperform',
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
    'collapse', 'crash', 'debt', 'deficit', 'default', 'layoff', 'liquidation', 'obsolete', 'penalty',
    'poor', 'shortage', 'shrink', 'strain', 'struggle', 'suspend', 'taper', 'tension', 'terminate',
    'threat', 'uncertain', 'underperform', 'unstable', 'vulnerability', 'volatile', 'writedown', 'unpredictable', 'crash'
]

neutral_words = [
    "report", "statement", "data", "meeting", "announcement", "update", "summary", "event", "discussion", "figure",
    'unchanged', 'neutral', 'maintain', 'expect', 'keep',
    'estimate', 'guidance', 'target', 'announce', 'report', 'quarter',
    'fiscal', 'year', 'slightly', 'slight', 'pretty', 'slow', 'prudent', 'insignificant', 'unsignificant',
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
    "optimistic market forecast", "profit margins rise", "revenue increase",
    "business revenue grows", "investor optimism increases", "positive economic outlook",
    "decrease losses", "reduce risk", "cut expenses", "lower costs", "diminish debt",
    "shrink deficit", "minimize exposure", "decline in expenses", "falling costs",
    "efficient cost structure", "strategic restructuring", "favorable refinancing",
    "streamlined operations", "cost optimization", "enhanced productivity", "decrease loss"
]

negative_phrases = [
    "market crash", "earnings decline", "economic slowdown", "investment drop", "increase loss"
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
    "analyst consensus", "market assessment", "industry benchmark", "comparative analysis",
    "financial metrics", "balance sheet items", "liquidity measures", "valuation model",
    "technical indicators", "market capitalization", "shareholder structure", "regulatory filing"
]

tricky_positives = [
# negative words with positive context
    "decrease in expenses", "falling interest rates", "declining unemployment",
    "reduced tax burden", "negative interest rates boosting economy", "shrinking deficit",
    "lower inflation", "slowing debt accumulation", "bearish signals ended",
    "downward adjustment of overvalued assets", "cutting excessive spending",
    "slumping prices benefit consumers", "halting unprofitable ventures",
    "collapse in production costs", "downgrade reversed", "decrease losses", "reduce risk",
    "weaker dollar strengthens exports", "hostile takeover improves efficiency",
    "credit rating downgrade was already priced in", "inflation higher than forecast but stabilizing",
    "defensive stocks outperforming", "inverted yield curve signaling recovery",
    "short interest decreasing", "positive leverage effect", "negative correlation with market decline",
    "buybacks during market downturn", "increased provisions for future losses",
    "higher capital requirements improving stability", "stricter regulations boosting confidence",
    "oil prices falling benefit airlines", "pharmaceutical patent expiry but generic opportunities",
    "retail inventory liquidation improving cash flow", "mining production cuts stabilizing prices",
    "agricultural oversupply lowering food inflation", "real estate cooling improving affordability",
    "currency devaluation boosting exports", "central bank intervention calming markets"
]
tricky_negatives = [
# positive words with negative context
    "growth in liabilities", "rising debt levels", "expanding trade deficit",
    "improving bankruptcy numbers", "strong decline", "accelerating downturn",
    "robust sell-off", "higher unemployment", "growing concerns", "surge in inflation",
    "boost to tax burden", "increased volatility", "positive cash burn rate", "increase deficit", "rise in debt"
    "effective cost cutting resulting in layoffs", "successful restructuring with job losses", "improve decline",
    "increase in bankruptcy", "decline in profits", "increase in losses", "fall in revenue",
    "surge in debts", "boost in negative returns", "surpass expectations but lose",
    "economic boom with high unemployment", "growth with rising inflation",
    "better than expected losses", "missed revenue targets but increased margins",
    "profits fell less than anticipated", "downward revision of upward estimates",
    "slow growth but stable outlook", "not as bad as feared", "could have been worse",
    "failed merger saves costs", "reduced guidance but positive long-term",
    "narrowly avoided bankruptcy through emergency funding", "default risk increases chances of buyout",
    "mark-to-market losses but improved underlying business", "debt refinancing at higher rates",
    "dividend cut to preserve cash", "positive carry trade unwinding", "orderly liquidation",
    "aggressive accounting write-downs", "goodwill impairment with tax benefits",
    "banks increasing loan loss reserves", "tech companies reducing headcount to boost margins",
    "energy transition costs but long-term efficiencies", "insurance premium increases offset claims",
    "healthcare cost inflation affecting insurance profits", "tariff increases impacting supply chains",
    "sovereign debt downgrade already expected"
]

intensifiers = [
    "much", "major", "significant", "substantial", "considerable", "notable",
    "great", "large", "strong", "tremendous", "enormous", "massive", "extensive",
    "marked", "pronounced", "immense", "meaningful", "huge", "hefty"
]

def intensify_score(score):
    if score > 0:
        return round(random.uniform(0.8, 0.97), 2)
    elif score < 0:
        return round(random.uniform(-0.97, -0.8), 2)
    return score


def pos_score(): return round(random.uniform(0.6, 0.8), 2)
def neg_score(): return round(random.uniform(-0.8, -0.6), 2)
def neu_score(): return round(random.uniform(-0.2, 0.2), 2)
def flip_score(score): return round(-score + random.uniform(-0.1, 0.1), 2)

negations = ["not", "no", "never", "neither", "n't", "hardly", "scarcely", "barely"]

dataset = []

for word in positive_words:
    dataset.append((word, pos_score()))
for word in negative_words:
    dataset.append((word, neg_score()))
for word in neutral_words:
    dataset.append((word, neu_score()))

for phrase in positive_phrases:
    dataset.append((phrase, pos_score()))
for phrase in negative_phrases:
    dataset.append((phrase, neg_score()))
for phrase in neutral_phrases:
    dataset.append((phrase, neu_score()))

positive_set = set(positive_words + positive_phrases)
negative_set = set(negative_words + negative_phrases)

intensified_dataset = []

for text, score in dataset:
    if text in positive_words or text in negative_words:
        for word in intensifiers:
            intensified_text = f"{word} {text}"
            intensified_score = intensify_score(score)
            intensified_dataset.append((intensified_text, intensified_score))

dataset.extend(intensified_dataset)

all_items = dataset.copy()
for text, score in all_items:
    if text in positive_words or text in negative_words:
        for neg in negations:
            if "n't" in neg:
                words = text.split()
                if len(words) > 1:
                    negated = f"{words[0]}{neg} {' '.join(words[1:])}"
                else:
                    negated = f"{text}{neg}"
            elif neg in ["neither", "nor"]:
                negated = f"{neg} {text} nor {text}"
            else:
                negated = f"{neg} {text}"
            dataset.append((negated, flip_score(score)))


def score_tricky_phrase(phrase):
    if phrase in tricky_positives:
        return pos_score()

    elif phrase in tricky_negatives:
        return neg_score()

    return neu_score()


for phrase in tricky_positives + tricky_negatives:
    score = score_tricky_phrase(phrase)
    dataset.append((phrase, score))

# for entry in dataset:
#     print(entry)
#
# all_items = dataset.copy()
# for text, score in all_items:
#     if any(neg in text.lower() for neg in negations):
#         continue
#
#     if score > 0.1:
#         for i, neg1 in enumerate(negations):
#             for neg2 in negations[i + 1:]:
#                 double_negated = f"{neg1} {neg2} {text}"
#                 new_score = score
#                 dataset.append((double_negated, round(new_score, 2)))
#
#     elif score < -0.1:
#         for i, neg1 in enumerate(negations):
#             for neg2 in negations[i + 1:]:
#                 double_negated = f"{neg1} {neg2} {text}"
#                 new_score = score
#                 dataset.append((double_negated, round(new_score, 2)))


print(dataset)
random.shuffle(dataset)

output_dir = "./sentiment_datasets"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "financial_sentiment_words_phrases_negations.csv")

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    #writer.writerow(["text", "score"])
    writer.writerows(dataset)