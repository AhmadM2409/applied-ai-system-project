# Model Card: Mood Machine

## 1. Model Overview

The Mood Machine is an applied AI mood-classification project for short text posts. It compares two implemented systems:

- A rule-based `MoodAnalyzer` in `mood_analyzer.py`
- A machine learning classifier in `ml_experiments.py` using `CountVectorizer` and Logistic Regression

The system predicts one of four dataset labels: `positive`, `negative`, `neutral`, or `mixed`. The rule-based model can also return `uncertain` as a guardrail output when confidence is too low.

The intended purpose is educational and experimental. It is designed to show how mood classification, confidence scoring, guardrails, and evaluation can work in a small reliability-focused AI system. It is not production-ready.

## 2. Data

The dataset is defined in `dataset.py` using `SAMPLE_POSTS` and `TRUE_LABELS`. It contains 24 short text examples with matching labels.

The dataset includes:

- Positive examples
- Negative examples
- Neutral examples
- Mixed-emotion examples
- Slang examples such as "Lowkey" and "No cap"
- Emoji-style text such as `:)` and `:(`
- Negation cases such as "not happy" and "not excited"
- Ambiguous or sarcastic examples such as "I absolutely love getting stuck in traffic"

Labels are limited to `positive`, `negative`, `neutral`, and `mixed`. Some labels are subjective, especially for sarcasm, negation, and mixed emotional statements.

## 3. Rule-Based Model Behavior

The rule-based model uses simple word matching. It preprocesses text by stripping whitespace, converting text to lowercase, and splitting on spaces. It then counts matched words from `POSITIVE_WORDS` and `NEGATIVE_WORDS`.

`score_text()` returns:

- `positive_score`
- `negative_score`
- `total_score`
- `tokens`
- `matched_positive_words`
- `matched_negative_words`

`get_confidence()` calculates confidence as:

`abs(positive_score - negative_score) / max(total_tokens, 1)`

`predict_label()` returns:

- `positive` when the positive score is higher
- `negative` when the negative score is higher
- `neutral` when no positive or negative words are found
- `mixed` when positive and negative scores are equal and both are greater than zero
- `uncertain` when confidence is below `0.2`

The rule-based model is transparent and easy to inspect, but it struggles with sarcasm, negation, punctuation, and subtle context.

## 4. ML Model Behavior

The ML model uses `CountVectorizer` to convert text into bag-of-words features. It then trains a Logistic Regression classifier on `SAMPLE_POSTS` and `TRUE_LABELS`.

The ML system includes `predict_with_confidence()`, which uses `model.predict_proba()` and returns:

- the predicted label
- the highest predicted probability as confidence

The ML model can learn patterns from the expanded dataset, including examples with negation and sarcasm. However, it is trained and evaluated on the same small dataset, so its reported accuracy should not be treated as real-world performance.

## 5. Evaluation Results

Evaluation is implemented in `evaluator.py`. The evaluator runs both systems on the labeled dataset, prints predictions with confidence scores, and computes accuracy and average confidence.

Known evaluation results:

- Rule-based accuracy: `0.50`
- Rule-based average confidence: `0.16`
- ML accuracy: `1.00`
- ML average confidence: `0.71`

The ML accuracy is training-set accuracy because the model is evaluated on the same examples it was trained on. This is not proof of real-world generalization.

## 6. Limitations and Biases

The dataset is small and manually labeled. It may not represent different writing styles, dialects, languages, age groups, or cultural contexts.

The rule-based model depends heavily on exact word matches. It does not understand grammar, deeper meaning, or context. For example, "not happy" can still match the positive word "happy" unless additional negation logic is added.

The ML model may overfit because it trains and evaluates on the same 24 examples. Its confidence values may look strong even when the model has not been tested on truly new data.

Both systems can misclassify emotionally sensitive text. Neither system should be used to make important decisions about a person's mental health or wellbeing.

## 7. Ethical Considerations

Mood classification can be sensitive because it involves interpreting personal language. A wrong prediction could misrepresent someone's feelings, especially in ambiguous, sarcastic, or distressed messages.

The system should be treated as a learning tool, not as a reliable emotional assessment system. Human review is important when predictions are used for reflection or evaluation.

Privacy should also be considered if personal text is analyzed. The current project uses local sample posts and does not include external storage or sharing.

## 8. Misuse Prevention

The rule-based guardrail returns `uncertain` when confidence is too low. This helps prevent the system from presenting weak predictions as reliable.

The evaluator reports confidence and accuracy together, which makes model behavior more transparent. The model card also documents limitations so users do not overstate the system's reliability.

This project should not be used for grading, diagnosis, surveillance, hiring, discipline, or other high-stakes decisions.

## 9. What Surprised Me During Reliability Testing

The rule-based system became more cautious after confidence scoring was added. Some predictions that previously returned a direct label changed to `uncertain` because the confidence score was low.

This showed that accuracy and reliability are not the same thing. A model can produce an output for every input, but a reliability layer helps show when the system should be less certain.

It was also surprising that the ML model reached perfect accuracy on the dataset. This result looked strong, but it mainly showed that evaluating on the training set can be misleading.

## 10. AI Collaboration Reflection

A helpful AI suggestion was using confidence scoring and `evaluator.py` to make reliability measurable. This improved the project by turning reliability into something visible through accuracy, average confidence, and per-example prediction logs.

A flawed AI suggestion was treating RAG as useful for this project. The system does not need retrieval from external documents because it is focused on local mood classification, rule logic, confidence scoring, and evaluation.

## 11. Ideas for Improvement

Future improvements could include:

- Add a separate test set instead of evaluating only on the training examples
- Improve preprocessing for punctuation, emojis, and repeated letters
- Add simple negation handling for phrases like "not happy"
- Add more examples from different writing styles and emotional contexts
- Track confusion between `mixed`, `neutral`, and `uncertain`
- Compare training-set accuracy with held-out test accuracy
- Refine confidence thresholds after testing on more data
