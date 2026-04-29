# The Mood Machine: Reliability-Focused Mood Classification

The Mood Machine is a short-text mood classification project that compares a transparent rule-based classifier with a small machine learning classifier. The final version adds confidence scoring, guardrails, and an evaluator script so predictions can be measured instead of only displayed.

This project is an applied AI learning system, not a production-ready emotional assessment tool.

## Original Project

Original project name: **The Mood Machine**.

The starter project introduced a simple text classifier for labeling short posts as `positive`, `negative`, `neutral`, or `mixed`. It began with a rule-based `MoodAnalyzer` and optional machine learning experiments using scikit-learn.

## Final System

The final system extends the starter lab into a reliability-focused applied AI project. It includes:

- An expanded mood dataset in `dataset.py`
- A rule-based `MoodAnalyzer` using word matching
- A machine learning classifier using `CountVectorizer` and Logistic Regression
- Confidence scoring for both systems
- A rule-based guardrail that returns `uncertain` for low-confidence predictions
- `evaluator.py` for accuracy and average-confidence reporting
- A system architecture diagram in `assets/system_architecture.md`
- A completed reflection and model card in `model_card.md`

AI feature selected: **Reliability / Testing System**.

## Architecture Overview

The system architecture is documented in `assets/system_architecture.md`.

At a high level, text input flows through two classification paths:

- Rule-based path: preprocessing, word matching, confidence scoring, and a low-confidence guardrail
- ML path: `CountVectorizer`, Logistic Regression, predicted label, and probability-based confidence

The evaluator compares both systems against the labeled dataset, prints prediction logs, and reports accuracy plus average confidence for human review.

## Repository Structure

```plaintext
dataset.py                    # Expanded word lists and labeled mood examples
mood_analyzer.py              # Rule-based classifier, confidence scoring, and guardrails
main.py                       # Rule-based evaluation and interactive demo
ml_experiments.py             # ML classifier and ML confidence prediction
evaluator.py                  # Reliability evaluation for both systems
model_card.md                 # Final model card and reflection
requirements.txt              # Python dependencies
assets/system_architecture.md # Mermaid system architecture diagram
```

## Setup

1. Create or activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The ML and evaluator scripts require `scikit-learn`.

## How to Run

Run the rule-based demo:

```bash
python main.py
```

Run the ML experiment:

```bash
python ml_experiments.py
```

Run the reliability evaluator:

```bash
python evaluator.py
```

## Sample Interactions

Example rule-based interactions:

```plaintext
Input: I am tired and overwhelmed
Output: negative

Input: This is fine
Output: neutral

Input: I love this class so much
Output: uncertain
```

The third example shows the guardrail behavior: the rule-based model detects a positive word, but its confidence is low because the sentence has several tokens and only one direct mood match.

## Design Decisions

- The rule-based model stays simple and interpretable so its behavior can be inspected.
- Confidence scoring was added to make prediction reliability visible.
- The `uncertain` guardrail prevents low-confidence rule-based predictions from being treated as strong labels.
- The ML model uses a small bag-of-words pipeline to provide a clear comparison with the rule-based approach.
- Evaluation reports both accuracy and average confidence because accuracy alone does not show how certain the systems are.

## Testing Summary

Known evaluator results:

- Rule-based accuracy: `0.50`
- Rule-based average confidence: `0.16`
- ML accuracy: `1.00`
- ML average confidence: `0.71`

The ML accuracy is training-set accuracy because the model is evaluated on the same examples it was trained on. It is not proof of real-world generalization.

## Limitations

- The dataset is small and manually labeled.
- The rule-based model depends on exact word matches.
- Sarcasm, negation, punctuation, and ambiguous emotional language remain difficult.
- The ML model may overfit because there is no separate test set yet.
- The system should not be used for high-stakes emotional, mental-health, academic, hiring, or disciplinary decisions.

## Reflection

This project showed that reliability is more than producing a label. Adding confidence scores, guardrails, and an evaluator made it easier to see where the system was uncertain and where accuracy numbers could be misleading. The strongest lesson was that the ML model's perfect score on the current dataset should be treated carefully because it reflects training-set performance, while the rule-based model is easier to inspect but less flexible.

## Loom Walkthrough

Loom video: [Add Loom walkthrough link here]
