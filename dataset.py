"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "loved",
    "loving",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "hopeful",
    "proud",
    "grateful",
    "calm",
    "peaceful",
    "confident",
    "joyful",
    "glad",
    "best",
    "fantastic",
    "perfect",
    "passed",
    "learned",
    "win",
    "winning",
    "yay",
    "lol",
    "vibes",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "stress",
    "hate",
    "boring",
    "anxious",
    "worried",
    "lonely",
    "overwhelmed",
    "frustrated",
    "annoyed",
    "miserable",
    "rough",
    "exhausted",
    "nervous",
    "crying",
    "pain",
    "fail",
    "failed",
    "ugh",
    "meh",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "I feel proud of my work today",
    "This project is going great",
    "Lowkey stressed but kind of proud of myself",
    "Highkey loving these chill vibes",
    "No cap this homework was awful",
    "I passed the quiz yay",
    "I am tired and overwhelmed",
    "The meeting starts at three",
    "Okay, sure, that was just perfect",
    "I absolutely love getting stuck in traffic",
    "Not bad, I guess",
    "I am not excited for tomorrow",
    "I feel calm and relaxed tonight",
    "I failed the test but learned a lot",
    "This is just another normal Tuesday",
    "I am crying but also laughing lol",
    "Best day ever :)",
    "I feel lonely tonight :(",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "positive",  # "I feel proud of my work today"
    "positive",  # "This project is going great"
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "positive",  # "Highkey loving these chill vibes"
    "negative",  # "No cap this homework was awful"
    "positive",  # "I passed the quiz yay"
    "negative",  # "I am tired and overwhelmed"
    "neutral",   # "The meeting starts at three"
    "negative",  # "Okay, sure, that was just perfect"
    "negative",  # "I absolutely love getting stuck in traffic"
    "mixed",     # "Not bad, I guess"
    "negative",  # "I am not excited for tomorrow"
    "positive",  # "I feel calm and relaxed tonight"
    "mixed",     # "I failed the test but learned a lot"
    "neutral",   # "This is just another normal Tuesday"
    "mixed",     # "I am crying but also laughing lol"
    "positive",  # "Best day ever :)"
    "negative",  # "I feel lonely tonight :("
]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
