import re

class PunctuationAnalyzer:
    def __init__(self):
        self.exaggerated_patterns = [
            r'!{2,}',     # >=2 exclamation marks
            r'\?{2,}',    # >=2 question marks
            r'\.{3,}',    # >=3 dots
            r'[,;:]{2,}'  # >=2 commas/semicolons/colons
        ]

    def count_exaggerated_punctuation(self, text):
        count = 0
        for pattern in self.exaggerated_patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        return count

    def get_punctuation_score(self, text, penalty_factor=1):
        if not text.strip():
            return 0.0

        num_exaggerated = self.count_exaggerated_punctuation(text)
        print(f"Exaggerated punctuation marks: {num_exaggerated}")

        num_words = len(text.split())
        if num_words == 0:
            return 0.0

        abuse_rate = num_exaggerated / num_words
        penalized_rate = abuse_rate * penalty_factor

        score = max(0.05, 1.0 - penalized_rate)
        return round(score, 3)

    def run_punctuation_analysis(self, text):
        return self.get_punctuation_score(text)

analyzer = PunctuationAnalyzer()

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
score = analyzer.run_punctuation_analysis(test_text)

print("Punctuation Credibility Score:", score)