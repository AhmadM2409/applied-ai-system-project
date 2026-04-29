# The Mood Machine

The Mood Machine is a simple text classifier that begins with a rule based approach and can optionally be extended with a small machine learning model. It tries to guess whether a short piece of text sounds **positive**, **negative**, **neutral**, or even **mixed** based on patterns in your data.

This lab gives you hands on experience with how basic systems work, where they break, and how different modeling choices affect fairness and accuracy. You will edit code, add data, run experiments, and write a short model card reflection.

## Final Project Direction

This project extends the original Mood Machine starter lab into a reliability-focused applied AI system.

The final system will improve mood classification by adding:
- a more specialized mood dataset
- improved rule-based classification logic
- machine learning comparison
- confidence scoring
- guardrails for uncertain or risky inputs
- structured evaluation/testing

Required AI feature selected: Reliability / Testing System.

Supporting enhancement: Specialized model behavior through improved data and classification rules.

## Reliability and Evaluation

This system includes a reliability layer to ensure predictions are not only generated, but also evaluated and explained.

Key additions:

- Confidence scoring:
  Both the rule-based and ML models return a confidence value for each prediction.
  - Rule-based confidence is based on the difference between positive and negative scores.
  - ML confidence is based on predicted probabilities.

- Guardrails:
  The rule-based model returns "uncertain" when confidence is too low, preventing unreliable predictions.

- Evaluation script:
  The project includes an evaluator script that:
  - runs both models on the dataset
  - prints predictions, true labels, and confidence scores
  - computes accuracy and average confidence

These additions allow the system to demonstrate reliability, transparency, and measurable performance rather than just producing outputs.

---

## Repo Structure

```plaintext
├── dataset.py         # Starter word lists and example posts (you will expand these)
├── mood_analyzer.py   # Rule based classifier with TODOs to improve
├── main.py            # Runs the rule based model and interactive demo
├── ml_experiments.py  # (New) A tiny ML classifier using scikit-learn
├── model_card.md      # Template to fill out after experimenting
└── requirements.txt   # Dependencies for optional ML exploration
```

---

## Getting Started

1. Open this folder in VS Code.
2. Make sure your Python environment is active.
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the rule-based starter:

    ```bash
    python main.py
    ```

If pieces of the analyzer are not implemented yet, you will see helpful errors that guide you to the TODOs.

To try the ML model later, run:

```bash
python ml_experiments.py
```

---

## What You Will Do

During this lab you will:

- Implement the missing parts of the rule based `MoodAnalyzer`.
- Add new positive and negative words.
- Expand the dataset with more posts, including slang, emojis, sarcasm, or mixed emotions.
- Observe unusual or incorrect predictions and think about why they happen.
- Train a tiny machine learning model and compare its behavior to your rule based system.
- Complete the model card with your findings about data, behavior, limitations, and improvements.
- The goal is to help you reason about how models behave, how data shapes them, and why even small design choices matter.

---

## Tips

- Start with preprocessing before updating scoring rules.
- When debugging, print tokens, scores, or intermediate choices.
- Ask an AI assistant to help create edge case posts or unusual wording.
- Try examples that mislead or confuse your model. Failure cases teach you the most.
