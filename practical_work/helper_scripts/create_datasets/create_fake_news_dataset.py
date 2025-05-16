import csv
import random
import os

import pandas as pd
import numpy as np

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

low_credibility_words = [
    "sources say", "rumor", "allegedly", "anonymous", "unofficial", "unconfirmed", "purported",
    "exclusive", "breaking", "shocking", "sensational", "bombshell", "secret", "insider",
    "revealed", "expose", "scandal", "controversy", "leaked", "unnamed", "confidential",
    "unverified", "speculated", "guessed", "hinted", "suggested", "might", "could", "may",
    "supposedly", "enormous", "massive", "tremendous", "huge", "meltdown", "catastrophe",
    "extraordinary", "unprecedented", "unlikely", "rare", "miracle", "magic", "revolutionary",
    "groundbreaking", "game-changing", "insiders", "sources close to", "conspiracy",
    "viral", "trending", "buzz", "explosive", "radical", "extreme", "mysterious", "hidden",
    "secret", "undisclosed", "controversial", "suspicious", "shady", "skeptical", "questionable",
    "clickbait", "misleading", "exaggerated", "fabricated", "distorted", "manipulated", "false",
    "fake", "hoax", "dubious", "unsubstantiated", "baseless", "unfounded", "gossip", "hearsay", "guru"
]

high_credibility_words = [
    "confirmed", "verified", "official", "according to", "reported by", "documented", "cited",
    "referenced", "sourced from", "evidenced", "proven", "demonstrated", "stated", "announced",
    "released", "published", "disclosed", "revealed by", "affirmed", "validated", "corroborated",
    "fact-checked", "authenticated", "substantiated", "established", "verified by experts",
    "peer-reviewed", "regulatory filing", "SEC filing", "annual report", "quarterly report",
    "earnings report", "press release", "financial statement", "10-K", "10-Q", "balance sheet",
    "income statement", "cash flow statement", "audited", "certified", "accredited", "licensed",
    "authorized", "reputable", "credible", "reliable", "trustworthy", "accurate", "precise",
    "measured", "calculated", "quantified", "estimated", "historical data", "statistical analysis",
    "research findings", "evidence-based", "empirical", "data-driven", "methodical", "systematic"
]

neutral_credibility_words = [
    "said", "reported", "noted", "mentioned", "indicated", "discussed", "commented", "addressed",
    "shared", "expressed", "stated", "explained", "described", "outlined", "highlighted",
    "presented", "suggested", "pointed out", "observed", "remarked", "added", "continued",
    "concluded", "summarized", "briefed", "informed", "updated", "advised", "notified",
    "communicated", "announced", "declared", "pronounced", "specified", "detailed", "clarified",
    "elaborated", "defined", "characterized", "identified", "recognized", "acknowledged"
]

low_credibility_phrases = [
    "anonymous sources claim", "rumors are circulating", "unconfirmed reports suggest", "act now", "before it is too late"
    "insiders reveal", "speculation grows about", "alleged documents show", "today only", "limited time"
    "unidentified officials say", "sources close to the situation", "leaked information indicates",
    "exclusive bombshell report", "shocking revelation", "unnamed experts warn", "I heard", "We heard", "This guy on Reddit said",
    "controversial theory suggests", "suspicious activity detected", "hidden agenda revealed", "Everyone’s talking about",
    "secret documents expose", "dramatic shift in", "mainstream media won't tell you",
    "what they don't want you to know", "the truth they're hiding", "too big to be coincidence",
    "connecting the dots reveals", "the story they're not telling", "major cover-up exposed",
    "guaranteed investment returns", "insider trading opportunities", "secret investment strategy",
    "stock market manipulation exposed", "financial experts shocked by", "markets on the brink of",
    "economic collapse imminent", "financial doomsday approaching", "get rich quick with",
    "bypassing financial regulations", "trillion-dollar industry secret", "banks don't want you to know",
    "financial elites panicking over", "money-making scheme revealed", "astronomical profits guaranteed",
    "financial miracle discovered", "unprecedented market opportunity", "secret wealth transfer"
]

high_credibility_phrases = [
    "according to audited financial statements", "verified by independent analysts",
    "official company announcement", "regulatory filing confirms", "peer-reviewed research shows",
    "as documented in quarterly reports", "multiple sources corroborate", "data from reliable sources indicates",
    "publicly available records show", "fact-checked by multiple organizations",
    "extensively researched analysis", "transparent methodology reveals", "consistent with historical patterns",
    "empirical evidence demonstrates", "systematic review indicates", "consensus among experts",
    "statistically significant findings", "replicated studies confirm", "longitudinal data shows",
    "cross-referenced from multiple sources", "published financial results indicate",
    "verified through official channels", "confirmed by regulatory authorities",
    "consistent with market fundamentals", "supported by comprehensive data",
    "quantitative analysis reveals", "verified earnings report shows", "official economic indicators suggest",
    "evidence-based projection indicates", "certified financial statements reveal",
    "independently audited results show", "verifiable market data indicates",
    "confirmed by company leadership", "documented in SEC filings"
]

neutral_credibility_phrases = [
    "quarterly earnings reported", "company statement issued", "market analysis suggests",
    "financial outlook updated", "investors responded to news", "trading volume increased",
    "share price fluctuated", "dividend announcement made", "merger discussions ongoing",
    "industry trends indicate", "sector performance varied", "economic indicators released",
    "inflation data published", "employment figures updated", "interest rates unchanged",
    "central bank decision announced", "trade balance reported", "consumer sentiment measured",
    "retail sales data shared", "housing market statistics", "manufacturing output reported",
    "service sector growth noted", "energy prices fluctuated", "commodity values adjusted",
    "currency exchange rates shifted", "bond yields moved", "credit ratings reviewed",
    "fiscal policy discussed", "monetary strategy outlined", "economic forecast presented"
]

hedging_phrases = [
    "might be", "could potentially", "perhaps indicates", "possibly suggests",
    "seems to imply", "appears to show", "allegedly involved in", "reportedly connected to",
    "rumored to be considering", "speculated to announce", "thought to be preparing",
    "believed to be planning", "expected by some to", "predicted by certain analysts",
    "estimated by unofficial sources", "hinted at by insiders", "suggested by anonymous sources",
    "implied by market movements", "indicated by unusual activity", "theorized by some experts"
]

precise_attribution = [
    "according to the quarterly financial report", "as stated by the CEO during the earnings call",
    "reported in the SEC filing dated", "confirmed by the Chief Financial Officer",
    "documented in the annual shareholder meeting", "presented in the audited financial statements",
    "disclosed in company press release number", "verified by external auditor",
    "published in the regulatory disclosure", "announced by the Board of Directors",
    "outlined in the investor presentation", "detailed in the Form 10-K",
    "specified in the merger agreement", "calculated based on public financial data",
    "derived from official government statistics", "compiled from mandatory disclosures"
]

sensationalist_phrases = [
    "financial disaster",
    "devastating losses", "market bloodbath", "economic nightmare",
    "financial apocalypse", "market collapse", "economic tsunami", "financial armageddon",
    "stock market freefall", "economic doomsday", "financial chaos", "market carnage",
    "economic implosion", "financial devastation", "market panic", "economic ruin",
    "explosive growth", "astronomical profits", "massive windfall", "unbelievable returns",
    "incredible opportunity", "spectacular gains", "extraordinary performance", "revolutionary product",
    "game-changing innovation", "paradigm-shifting development"
]

balanced_phrases = [
    "mixed quarterly results", "both positive and negative factors", "advantages and limitations",
    "strengths and weaknesses", "opportunities and challenges", "benefits and drawbacks",
    "profits in some sectors, losses in others", "promising yet cautious outlook",
    "moderate growth with concerns", "partial success with ongoing issues",
    "progress despite obstacles", "improvement with persistent challenges",
    "recovery alongside continued risks", "gains tempered by setbacks",
    "optimistic forecast with caveats", "encouraging signs with notable exceptions",
    "strategic vision with implementation hurdles", "competitive advantage with market constraints",
    "innovative approach with adoption challenges", "cost savings offset by new expenses"
]

tricky_low_credibility = [
    "official-looking report reveals shocking truth", "expert analysis proves conspiracy",
    "scientific study supports controversial claim", "leaked official document confirms suspicions",
    "authoritative source exposes massive scandal", "verified insider reveals shocking secrets",
    "documented evidence of unprecedented cover-up", "statistical analysis proves manipulation",
    "peer-reviewed study challenges established facts", "comprehensive investigation uncovers conspiracy",
    "independent research confirms fringe theory", "methodical analysis reveals shocking pattern",
    "systematic examination exposes hidden agenda", "reliable sources confirm spectacular discovery",
    "rigorous testing reveals miracle solution", "detailed documentation of secret scheme"
]


def lemmatize_phrase(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.lemma_.strip()])


def low_score(): return round(random.uniform(0.01, 0.3), 2)


def medium_low_score(): return round(random.uniform(0.3, 0.45), 2)


def neutral_score(): return round(random.uniform(0.45, 0.55), 2)


def medium_high_score(): return round(random.uniform(0.55, 0.7), 2)


def high_score(): return round(random.uniform(0.7, 0.99), 2)


dataset = []

for word in low_credibility_words:
    word = lemmatize_phrase(word)
    for _ in range(10):
        dataset.append((word, low_score()))

for word in high_credibility_words:
    word = lemmatize_phrase(word)
    for _ in range(10):
        dataset.append((word, high_score()))

# for word in neutral_credibility_words:
#     word = lemmatize_phrase(word)
#     for _ in range(5):
#         dataset.append((word, neutral_score()))

for phrase in low_credibility_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, low_score()))

for phrase in high_credibility_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, high_score()))

# for phrase in neutral_credibility_phrases:
#     phrase = lemmatize_phrase(phrase)
#     for _ in range(5):
#         dataset.append((phrase, neutral_score()))

for phrase in hedging_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(5):
        dataset.append((phrase, medium_low_score()))

# for phrase in precise_attribution:
#     phrase = lemmatize_phrase(phrase)
#     for _ in range(5):
#         dataset.append((phrase, medium_high_score()))

for phrase in sensationalist_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, low_score()))

for phrase in balanced_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, high_score()))

for phrase in tricky_low_credibility:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, low_score()))

combined_examples = []

low_sources = [
    "anonymous sources claim", "rumors suggest", "unconfirmed reports indicate",
    "alleged insiders reveal", "social media users speculate", "an unverified blog states",
    "a viral post asserts", "unofficial channels suggest", "unreliable sources indicate",
    "a whistleblower allegedly revealed", "according to an unnamed source",
    "sources speaking on condition of anonymity"
]

high_claims = [
    "quarterly earnings exceeded expectations", "a new strategic partnership has been finalized",
    "regulatory approval has been granted", "significant cost reductions have been achieved",
    "market share has increased substantially", "a major technological breakthrough has occurred",
    "expansion into new markets is successful", "debt restructuring has been completed favorably",
    "a substantial acquisition has been finalized", "production efficiency has improved dramatically"
]

for source in low_sources:
    for claim in high_claims:
        combined = f"{source} that {claim}"
        combined = lemmatize_phrase(combined)
        dataset.append((combined, medium_low_score()))

high_sources = [
    "according to the official report", "as stated in the press release",
    "the audited financial statement shows", "regulatory filings confirm",
    "the CEO announced during the meeting", "the board of directors disclosed",
    "the independent audit revealed", "as documented in SEC filings",
    "verified data indicates", "company documents show"
]

low_claims = [
    "unprecedented market disruption is imminent", "a revolutionary product will change the industry forever",
    "profits will increase exponentially", "competitors are on the verge of collapse",
    "the stock price could triple within months", "a miraculous turnaround has occurred",
    "financial problems have completely disappeared", "market dominance is guaranteed",
    "all risks have been eliminated", "success is absolutely certain"
]

for source in high_sources:
    for claim in low_claims:
        combined = f"{source} that {claim}"
        combined = lemmatize_phrase(combined)
        dataset.append((combined, medium_high_score()))

low_credibility_headlines = [
    f"BREAKING: {random.choice(['Secret', 'Shocking', 'Explosive', 'Bombshell'])} Report Reveals {random.choice(['Massive', 'Huge', 'Enormous'])} {random.choice(['Scandal', 'Cover-up', 'Conspiracy'])} at Major Bank",
    f"{random.choice(['Insiders', 'Anonymous Sources', 'Whistleblowers'])} Claim {random.choice(['Leading', 'Major', 'Prominent'])} Tech Company is Hiding Significant Losses",
    f"EXCLUSIVE: {random.choice(['Leaked', 'Confidential', 'Secret'])} Documents Show {random.choice(['Impending', 'Imminent', 'Looming'])} Financial Collapse",
    f"Market Guru Predicts {random.choice(['Spectacular', 'Incredible', 'Unprecedented'])} 500% Gains in This {random.choice(['Unknown', 'Secret', 'Hidden'])} Stock",
    f"Financial Elite {random.choice(['Panicking', 'Worried', 'Terrified'])} as {random.choice(['Revolutionary', 'Game-changing', 'Disruptive'])} New Cryptocurrency {random.choice(['Soars', 'Explodes', 'Skyrockets'])}"
]

high_credibility_headlines = [
    f"{random.choice(['Annual', 'Quarterly', 'Monthly'])} Economic Report Shows {random.choice(['Moderate', 'Steady', 'Consistent'])} Growth in Manufacturing Sector",
    f"{random.choice(['Federal Reserve', 'SEC', 'Treasury Department'])} Announces {random.choice(['New', 'Updated', 'Revised'])} Financial Regulations After Public Consultation",
    f"Company Reports {random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} Earnings: Revenue {random.choice(['Up', 'Down', 'Flat'])} 3.2%, Expenses {random.choice(['Increased', 'Decreased', 'Stable'])} by 2.1%",
    f"International Trade Organization Data Shows {random.choice(['Improvement', 'Decline', 'Stability'])} in Cross-Border Logistics Efficiency",
    f"Market Analysis: {random.choice(['Sector', 'Industry', 'Market'])} Performance Varies as {random.choice(['Economic Indicators', 'Leading Indicators', 'Financial Metrics'])} Show Mixed Results"
]

for headline in low_credibility_headlines:
    headline = lemmatize_phrase(headline)
    dataset.append((headline, low_score()))

# for headline in high_credibility_headlines:
#     headline = lemmatize_phrase(headline)
#     dataset.append((headline, high_score()))

low_credibility_stories = [
    f"{random.choice(low_sources)} that a {random.choice(['leading', 'major', 'prominent'])} tech company is hiding significant losses, with {random.choice(['insiders', 'sources', 'experts'])} hinting at an impending bankruptcy.",
    f"{random.choice(['Shocking', 'Exclusive', 'Breaking'])} report reveals that investors could make {random.choice(['enormous', 'massive', 'spectacular'])} profits with this {random.choice(['little-known', 'secret', 'overlooked'])} investment strategy.",
    f"Market {random.choice(['guru', 'wizard', 'genius'])} predicts that the stock market will either {random.choice(['crash spectacularly', 'soar to unprecedented heights', 'experience extreme volatility'])} in the coming weeks.",
    f"{random.choice(['Controversial', 'Mysterious', 'Shadowy'])} financial group claims to have discovered a {random.choice(['foolproof', 'guaranteed', 'perfect'])} method to predict market movements with 100% accuracy.",
    f"The financial establishment doesn't want you to know about this {random.choice(['revolutionary', 'game-changing', 'groundbreaking'])} investment that could {random.choice(['make you rich overnight', 'transform your financial future', 'guarantee enormous returns'])}"
]

high_credibility_stories = [
    f"{random.choice(high_sources)}, the company reported a {random.choice(['5.3%', '2.7%', '3.9%'])} increase in quarterly revenue, while operating expenses {random.choice(['rose by 2.1%', 'fell by 1.8%', 'remained stable'])}, compared to the same period last year.",
    f"According to the {random.choice(['annual financial report', 'audited statements', 'regulatory filing'])}, the bank increased its loan loss provisions by {random.choice(['12%', '8%', '15%'])} in anticipation of economic headwinds in the coming fiscal year.",
    f"The {random.choice(['Federal Reserve', 'Central Bank', 'Treasury'])} announced a {random.choice(['25', '50', '75'])} basis point {random.choice(['increase', 'decrease', 'adjustment'])} to interest rates, citing {random.choice(['inflation concerns', 'employment data', 'economic indicators'])} as the primary factor.",
    f"Industry analysis based on {random.choice(['market data', 'sector performance', 'financial metrics'])} shows that the {random.choice(['technology', 'healthcare', 'energy'])} sector has outperformed the broader market by {random.choice(['3.7%', '2.9%', '4.2%'])} year-to-date.",
    f"A {random.choice(['comprehensive study', 'detailed analysis', 'systematic review'])} of economic indicators suggests that {random.choice(['consumer spending', 'business investment', 'export activity'])} may {random.choice(['increase moderately', 'decline slightly', 'remain stable'])} in the coming quarter."
]

for story in low_credibility_stories:
    story = lemmatize_phrase(story)
    dataset.append((story, low_score()))

# for story in high_credibility_stories:
#     story = lemmatize_phrase(story)
#     dataset.append((story, high_score()))

print(f"Dataset size: {len(dataset)}")
random.shuffle(dataset)

output_dir = "./credibility_datasets"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "custom_financial_news_credibility.csv")

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "credibility_score"])
    writer.writerows(dataset)

print(f"Dataset saved to {output_file}")

sample_size = min(20, len(dataset))
sample_data = random.sample(dataset, sample_size)
sample_df = pd.DataFrame(sample_data, columns=["text", "credibility_score"])
sample_df = sample_df.sort_values(by="credibility_score")

print("\nSample of the dataset:")
print(sample_df)