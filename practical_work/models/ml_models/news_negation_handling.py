import numpy as np
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


class NegationHandlingTransformer:
    """
    A post-processing layer that handles negation and double negation in financial text
    for transformer-based sentiment analysis models.
    """

    def __init__(self, base_model, tokenizer, device, threshold=0.25):
        """
        Initialize the negation handler with the base transformer model.

        Args:
            base_model: The BERT model for sentiment classification
            tokenizer: The tokenizer used with the model
            device: The device (cuda/cpu) for computation
            threshold: Confidence threshold for applying negation adjustment
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.negation_words = {
            "not", "n't", "no", "never", "neither", "nor", "none", "cannot",
            "didn't", "doesn't", "don't", "isn't", "wasn't", "aren't", "weren't",
            "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't",
            "without", "nobody", "nothing", "nowhere", "cease", "discontinue", "halt"
        }

        self.scope_terminators = {
            "but", "however", "nevertheless", "yet", "although", "though",
            ".", ",", ";", ":", "!", "?", "despite", "in spite", "regardless",
            "notwithstanding"
        }

        self.positive_financial_terms = {
            "growth", "profit", "gain", "rise", "increase", "improve", "recovery",
            "upward", "bullish", "outperform", "exceed", "uptrend", "rally",
            "strong", "positive", "enhance", "climb", "advantage", "opportunity",
            "dividend", "upgrade", "surpass", "momentum", "record", "historic",
            "higher", "beat", "above", "jump", "boom", "soar", "surge"
        }

        self.negative_financial_terms = {
            "loss", "decline", "drop", "fall", "decrease", "worsen", "bearish",
            "downward", "underperform", "below", "downtrend", "weak", "negative",
            "downgrade", "concern", "risk", "volatility", "uncertainty", "debt",
            "miss", "short", "lower", "reduce", "slump", "crash", "tumble", "plunge",
            "crisis", "recession", "inflation", "descending", "contraction"
        }

    def detect_negation_scope(self, text):
        """
        Identifies words affected by negation in financial text.
        Returns a set of words that should have their sentiment adjustments.
        """
        tokens = word_tokenize(text.lower())
        negated_words = set()

        negation_active = False
        negation_distance = 0
        max_negation_distance = 10

        for i, token in enumerate(tokens):
            if token in self.negation_words or any(neg in token for neg in ["n't", "not"]):
                negation_active = True
                negation_distance = 0
                continue

            if negation_active:
                negation_distance += 1

                if (token in self.scope_terminators or
                        negation_distance > max_negation_distance):
                    negation_active = False
                else:
                    if token not in self.stop_words:
                        negated_words.add(token)
                        negated_words.add(self.lemmatizer.lemmatize(token))

        return negated_words

    def contains_double_negation(self, text):
        """
        Detect if text contains double negation, especially involving financial terms.
        """
        negated_words = self.detect_negation_scope(text)

        for word in negated_words:
            lemma = self.lemmatizer.lemmatize(word)
            if lemma in self.negative_financial_terms:
                return True, word

        double_neg_patterns = [
            r"didn't\s+decline", r"didn't\s+decrease", r"didn't\s+drop", r"didn't\s+fall",
            r"not\s+descending", r"not\s+bearish", r"not\s+negative", r"didn't\s+lose",
            r"no\s+longer\s+declining", r"stopped\s+falling", r"halted\s+its\s+decline",
            r"didn't\s+keep\s+its\s+descending", r"not\s+continue\s+to\s+fall",
            r"reversed\s+the\s+downward", r"didn't\s+maintain\s+.*\s+downturn"
        ]

        for pattern in double_neg_patterns:
            if re.search(pattern, text.lower()):
                return True, pattern

        return False, None

    def predict_sentiment_with_negation_handling(self, text, max_length=64):
        """
        Predict sentiment with enhanced handling of negation and double negation.
        Returns the original and adjusted predictions.
        """
        self.base_model.eval()

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.base_model(**inputs)

        logits = outputs[0]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        original_pred_class = torch.argmax(probabilities, dim=-1).item()
        original_probs = probabilities.cpu().numpy()[0]

        has_double_neg, pattern = self.contains_double_negation(text)

        if not has_double_neg or max(original_probs) > 0.9:
            sentiment_classes = ["neutral", "positive", "negative"]
            return {
                'original_class': original_pred_class,
                'original_class_name': sentiment_classes[original_pred_class],
                'original_probabilities': original_probs,
                'adjusted_class': original_pred_class,
                'adjusted_class_name': sentiment_classes[original_pred_class],
                'adjusted_probabilities': original_probs,
                'adjustment_applied': False,
                'adjustment_reason': 'No double negation detected or high confidence'
            }

        adjusted_probs = original_probs.copy()

        if original_pred_class == 2:  # Negative
            # Transfer probability from negative to positive
            transfer_amount = adjusted_probs[2] * 0.7
            adjusted_probs[2] -= transfer_amount
            adjusted_probs[1] += transfer_amount  # Add to positive

            adjustment_reason = f"Double negation detected: {pattern}"

        elif original_pred_class == 0 and has_double_neg:  # Neutral
            transfer_amount = adjusted_probs[0] * 0.4
            adjusted_probs[0] -= transfer_amount
            adjusted_probs[1] += transfer_amount  # Add to positive

            adjustment_reason = f"Double negation of negative term detected: {pattern}"

        else:
            adjustment_reason = "No adjustment needed for this prediction type"

        adjusted_pred_class = np.argmax(adjusted_probs)

        sentiment_classes = ["neutral", "positive", "negative"]

        return {
            'original_class': original_pred_class,
            'original_class_name': sentiment_classes[original_pred_class],
            'original_probabilities': original_probs,
            'adjusted_class': adjusted_pred_class,
            'adjusted_class_name': sentiment_classes[adjusted_pred_class],
            'adjusted_probabilities': adjusted_probs,
            'adjustment_applied': True,
            'adjustment_reason': adjustment_reason
        }


def apply_negation_handling(model, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Create and return a negation-aware prediction function to wrap the model.
    """
    negation_handler = NegationHandlingTransformer(model, tokenizer, device)

    def predict_with_negation_handling(text, max_length=64):
        return negation_handler.predict_sentiment_with_negation_handling(text, max_length)

    return predict_with_negation_handling