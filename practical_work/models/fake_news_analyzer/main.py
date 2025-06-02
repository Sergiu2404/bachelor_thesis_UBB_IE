from domain_reliability_analyzer import NewsReliabilityChecker
from punctuation_analyzer import PunctuationAnalyzer
from grammar_analyzer import GrammarAnalyzer
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

text = "Breaking news: Some of the profit of the company will be invested to build a new Nuclear Reactor. [www.cnn.com]"

start = time.time()


def analyze_reliability():
    reliability_analyzer = NewsReliabilityChecker()
    return reliability_analyzer.get_reliability_score(text)


def analyze_punctuation():
    punctuation_analyzer = PunctuationAnalyzer()
    return punctuation_analyzer.run_punctuation_analysis(text)


def analyze_grammar():
    grammar_analyzer = GrammarAnalyzer()
    return grammar_analyzer.run_grammar_analysis(text)


results = {}

with ThreadPoolExecutor(max_workers=3) as executor:
    grammar_future = executor.submit(analyze_grammar)
    reliability_future = executor.submit(analyze_reliability)
    punctuation_future = executor.submit(analyze_punctuation)

    results["grammar"] = grammar_future.result()
    results["source"] = reliability_future.result()
    results["punctuation"] = punctuation_future.result()

for name in ["grammar", "source", "punctuation"]:
    print(f"{name}: {results[name]}")

print(f"it took {time.time() - start} to execute")