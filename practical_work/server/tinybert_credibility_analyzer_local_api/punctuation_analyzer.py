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

# analyzer = PunctuationAnalyzer()
#
# test_text = "Wait... what are you doing??!!! This is crazy!! The current article has got more credibility now..."
# score = analyzer.run_punctuation_analysis(test_text)
#
# print("Punctuation Credibility Score:", score)