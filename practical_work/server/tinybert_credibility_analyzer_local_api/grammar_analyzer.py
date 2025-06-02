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
#
# analyzer = GrammarAnalyzer()
#
# test_text = "This is a bad sentence it have many mistake. He go to school everyday. Do you think this is a well analyzed sentence?"
# score = analyzer.run_grammar_analysis(test_text)
#
# print("Grammar Credibility Score:", score)