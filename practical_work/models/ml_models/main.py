import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure necessary NLTK resources are available
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

class FinancialNewsAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, word):
        """Convert POS tag to format used by WordNetLemmatizer"""
        tag = pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if not found

    def preprocess_text(self, text):
        """Tokenize, lemmatize, and return preprocessed text"""
        text = text.lower()
        tokens = word_tokenize(text)
        print(tokens)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens]
        return ' '.join(lemmatized_tokens)

# Example usage
analyzer = FinancialNewsAnalyzer()
sample_text = "strength don't meet expectations"
processed_text = analyzer.preprocess_text(sample_text)
print(processed_text)  # Expected output: "the company grow rapidly and be now grow with strong grow."
