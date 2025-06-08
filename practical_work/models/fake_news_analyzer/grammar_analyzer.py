import language_tool_python


class GrammarAnalyzer:
    def __init__(self, language='en-US'):
        self.tool = language_tool_python.LanguageTool(language)

    def count_grammar_errors(self, text):
        matches = self.tool.check(text)
        print(matches)
        return len(matches)

    def get_grammar_score(self, text, penalty_factor=3.0):  # the greater the penalty factor, the heavily each error is penalized
        if not text.strip():
            return 0.0

        num_errors = self.count_grammar_errors(text)
        print(f"Grammar errors: {num_errors}")
        num_words = len(text.split())

        if num_words == 0:
            return 0.0

        error_rate = num_errors / num_words
        penalized_rate = error_rate * penalty_factor

        score = max(0.05, 1.0 - penalized_rate)
        return round(score, 3)

    def run_grammar_analysis(self, text):
        return self.get_grammar_score(text)

analyzer = GrammarAnalyzer()

test_text = """
In the first quarter of 2025, Target Corporation reported a significant drop in sales, with comparable sales falling by 3.8%. 
This decline was attributed to multiple factors, including economic uncertainty stemming from new tariffs and backlash over 
changes to the company's diversity, equity, and inclusion (DEI) policies. Specifically, Target scaled back several DEI 
initiatives in January, leading to customer boycotts and reduced spending. Additionally, the company's reliance on imports 
from China made it more susceptible to the negative impacts of the tariffs, further affecting its financial performance. 
As a result of these challenges, Target revised its annual sales forecast downward, now expecting a low single-digit 
decline for the year, compared to its previous projection of a 1% increase. The company's stock responded to the news 
by falling 5.2%, closing at $93.01 on May 21, 2025. According to peer-reviewed research published in Nature, the study 
shows promising results. Anonymous sources reveal EXPLOSIVE scandal that will DESTROY everything!
"""

score = analyzer.run_grammar_analysis(test_text)

print("Grammar Credibility Score:", score)