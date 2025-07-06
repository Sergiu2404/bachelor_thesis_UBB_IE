# import language_tool_python
#
#
# class GrammarAnalyzer:
#     def __init__(self, language='en-US'):
#         self.tool = language_tool_python.LanguageTool(language)
#
#     def count_grammar_errors(self, text):
#         matches = self.tool.check(text)
#         print(matches)
#         return len(matches)
#
#     def get_grammar_score(self, text, penalty_factor=3.0):  # the greater the penalty factor, the heavily each error is penalized
#         if not text.strip():
#             return 0.0
#
#         num_errors = self.count_grammar_errors(text)
#         print(f"Grammar errors: {num_errors}")
#         num_words = len(text.split())
#
#         if num_words == 0:
#             return 0.0
#
#         error_rate = num_errors / num_words
#         penalized_rate = error_rate * penalty_factor
#
#         score = max(0.05, 1.0 - penalized_rate)
#         return round(score, 3)
#
#     def run_grammar_analysis(self, text):
#         return self.get_grammar_score(text)




import language_tool_python
import re
import enchant

class GrammarAnalyzer:
    def __init__(self, language='en-US'):
        self.tool = language_tool_python.LanguageTool(language)
        self.spell_checker = enchant.Dict("en_US")

    def count_grammar_errors(self, text):
        matches = self.tool.check(text)
        print(f"\nGrammar Matches ({len(matches)}):")
        for m in matches:
            print(f"Rule: {m.ruleId}, Message: {m.message}, Error: '{text[m.offset:m.offset+m.errorLength]}'")
        return matches

    def count_gibberish_words(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        gibberish_words = [
            w for w in words if (
                len(w) > 12 and not self.spell_checker.check(w)
            )
        ]
        print(f"\nGibberish Words Detected ({len(gibberish_words)}): {gibberish_words}")
        return gibberish_words

    def get_grammar_score(self, text, penalty_factor=3.0, char_penalty_factor=0.3, gibberish_penalty=0.8):
        if not text.strip():
            return 0.0

        matches = self.count_grammar_errors(text)
        num_errors = len(matches)
        total_error_chars = sum([m.errorLength for m in matches if m.errorLength])

        num_words = len(text.split())
        num_chars = len(text)
        gibberish_words = self.count_gibberish_words(text)
        gibberish_count = len(gibberish_words)
        gibberish_char_count = sum(len(w) for w in gibberish_words)

        if num_words == 0 or num_chars == 0:
            return 0.0

        error_rate = num_errors / num_words
        char_error_rate = total_error_chars / num_chars
        gibberish_rate = gibberish_count / num_words
        gibberish_char_ratio = gibberish_char_count / num_chars

        penalized_rate = (
               (error_rate * penalty_factor)
            + (char_error_rate * char_penalty_factor)
            + ((gibberish_rate + gibberish_char_ratio) * gibberish_penalty)
        )

        score = max(0.05, 1.0 - penalized_rate)
        print(f"\nScore Components:\n- Errors: {num_errors}, Words: {num_words}, Error Rate: {error_rate:.3f}\n"
              f"Error Chars: {total_error_chars}, Char Error Rate: {char_error_rate:.3f}\n"
              f"Gibberish Words: {gibberish_count}, Rate: {gibberish_rate:.3f}, Char Ratio: {gibberish_char_ratio:.3f}\n"
              f"Final Score: {score:.3f}")
        return round(score, 3)

    def run_grammar_analysis(self, text):
        return self.get_grammar_score(text)
