# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> Dict[str, object]:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        matched_positive_words: List[str] = []
        matched_negative_words: List[str] = []

        for token in tokens:
            if token in self.positive_words:
                matched_positive_words.append(token)
            if token in self.negative_words:
                matched_negative_words.append(token)

        positive_score = len(matched_positive_words)
        negative_score = len(matched_negative_words)
        total_score = positive_score - negative_score

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "total_score": total_score,
            "tokens": tokens,
            "matched_positive_words": matched_positive_words,
            "matched_negative_words": matched_negative_words,
        }

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def get_confidence(self, text: str) -> float:
        """
        Estimate confidence from the gap between positive and negative scores.
        """
        scores = self.score_text(text)
        positive_score = scores["positive_score"]
        negative_score = scores["negative_score"]
        total_tokens = len(scores["tokens"])

        confidence = abs(positive_score - negative_score) / max(total_tokens, 1)
        return min(max(float(confidence), 0.0), 1.0)

    def predict_label(self, text: str, return_confidence: bool = False):
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        scores = self.score_text(text)
        positive_score = scores["positive_score"]
        negative_score = scores["negative_score"]
        confidence = self.get_confidence(text)

        if positive_score == 0 and negative_score == 0:
            label = "neutral"
        elif confidence < 0.2:
            label = "uncertain"
        elif positive_score > negative_score:
            label = "positive"
        elif negative_score > positive_score:
            label = "negative"
        else:
            label = "mixed"

        if return_confidence:
            return label, confidence
        return label

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
