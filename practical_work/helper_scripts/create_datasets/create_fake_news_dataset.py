import csv
import random
import os

import pandas as pd
import numpy as np

import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

low_credibility_words = [
    "rumor", "allegedly", "anonymous", "unofficial", "unconfirmed", "purported",
    "exclusive", "breaking", "shocking", "sensational", "bombshell", "secret", "insider",
    "revealed", "expose", "scandal", "controversy", "leaked", "unnamed", "confidential",
    "unverified", "speculated", "guessed", "hinted", "suggested", "might", "could", "may",
    "supposedly", "enormous", "massive", "tremendous", "huge", "meltdown", "catastrophe", "leaked",
    "extraordinary", "unprecedented", "unlikely", "rare", "miracle", "magic", "revolutionary",
    "groundbreaking", "game-changing", "insiders", "sources close to", "conspiracy", "unknown", "unchecked"
    "viral", "trending", "buzz", "explosive", "radical", "extreme", "mysterious", "hidden",
    "secret", "undisclosed", "controversial", "suspicious", "shady", "skeptical", "questionable",
    "clickbait", "misleading", "exaggerated", "fabricated", "distorted", "manipulated", "false",
    "fake", "hoax", "dubious", "unsubstantiated", "baseless", "unfounded", "gossip", "hearsay", "guru",
    "anonymous", "leaked", "unnamed", "unconfirmed", "allegedly", "purported", "unverified", "unauthenticated",
    "disputed", "fabricated", "unofficial", "dubious", "baseless", "speculative", "shady", "questionable",
    "rumored", "unacknowledged", "unsupported", "hidden", "secret", "obscure", "clandestine", "forbidden",
    "banned", "censored", "restricted", "blacklisted", "shadowy", "mysterious", "covert", "illicit", "plausible",
    "assumed", "suggested", "hinted", "whispered", "uncertain", "unbacked", "guessed", "implied", "estimated",
    "contested", "unauthorized", "invisible", "cryptic", "hazy", "vague", "unknown", "obscured", "blurred",
    "hypothetical", "fictional", "theoretical", "provisional", "tentative", "inconclusive", "ambiguous", "allegoric",
    "imaginary", "rumored", "miraculous", "mythical", "misleading", "exaggerated", "clickbait", "viral",
    "controversial", "divisive", "manipulated", "twisted", "deceptive", "falsified", "tampered", "staged", "hoax",
    "scam", "phony", "bogus", "pseudo", "false", "fraudulent", "counterfeit", "planted", "setup", "scheme",
    "gimmick", "ruse", "plot", "ploy", "conspiracy", "misrepresented", "outlandish", "unrealistic", "absurd",
    "fantastical", "unbelievable", "fabrication", "invention"
]

nouns = [
    "sources", "experts", "reports", "claims", "documents", "evidence", "leaks", "stories", "accounts", "statements",
    "information", "theory", "allegations", "channel", "witness", "post", "comment", "headline", "message", "statement"
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
    "secret documents expose", "dramatic shift in", "mainstream media won't tell you", "unknown sources",
    "what they don't want you to know", "the truth they're hiding", "too big to be coincidence",
    "connecting the dots reveals", "the story they're not telling", "major cover-up exposed",
    "guaranteed investment returns", "insider trading opportunities", "secret investment strategy",
    "stock market manipulation exposed", "financial experts shocked by", "markets on the brink of",
    "economic collapse imminent", "financial doomsday approaching", "get rich quick with",
    "bypassing financial regulations", "trillion-dollar industry secret", "banks don't want you to know",
    "financial elites panicking over", "money-making scheme revealed", "astronomical profits guaranteed",
    "financial miracle discovered", "unprecedented market opportunity", "secret wealth transfer", "sources say"
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


def low_score(): return round(random.uniform(0.01, 0.2), 2)


def medium_low_score(): return round(random.uniform(0.2, 0.45), 2)


def neutral_score(): return round(random.uniform(0.45, 0.55), 2)


def medium_high_score(): return round(random.uniform(0.55, 0.8), 2)


def high_score(): return round(random.uniform(0.8, 0.99), 2)


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

# for phrase in high_credibility_phrases:
#     phrase = lemmatize_phrase(phrase)
#     for _ in range(10):
#         dataset.append((phrase, high_score()))

# for phrase in neutral_credibility_phrases:
#     phrase = lemmatize_phrase(phrase)
#     for _ in range(5):
#         dataset.append((phrase, neutral_score()))

# for phrase in precise_attribution:
#     phrase = lemmatize_phrase(phrase)
#     for _ in range(5):
#         dataset.append((phrase, medium_high_score()))

for phrase in sensationalist_phrases:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, low_score()))

for phrase in tricky_low_credibility:
    phrase = lemmatize_phrase(phrase)
    for _ in range(10):
        dataset.append((phrase, low_score()))

for attr in set(low_credibility_words):
    for noun in nouns:
        phrase = lemmatize_phrase(f"{attr} {noun}")
        for _ in range(3):
            dataset.append((phrase, low_score()))

for phrase in neutral_credibility_phrases:
    for attr in list(set(low_credibility_words))[:10]:
        modified_phrase = lemmatize_phrase(f"{attr} {phrase}")
        dataset.append((modified_phrase, low_score()))

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