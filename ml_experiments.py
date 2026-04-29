# ml_experiments.py
"""
Simple ML experiments for the Mood Machine lab.

This file uses a "real" machine learning library (scikit-learn)
to train a tiny text classifier on the same SAMPLE_POSTS and
TRUE_LABELS that you use with the rule based model.
"""

import argparse
from collections import Counter
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loader import load_external_dataset
from dataset import SAMPLE_POSTS, TRUE_LABELS


def safe_for_console(value) -> str:
    """
    Convert text to a form that Windows terminals can print safely.
    """
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> Tuple[CountVectorizer, LogisticRegression]:
    """
    Train a simple text classifier using bag of words features
    and logistic regression.

    Steps:
      1. Convert the texts into numeric vectors using CountVectorizer.
      2. Fit a LogisticRegression model on those vectors and labels.

    Returns:
      (vectorizer, model)
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def load_training_data(
    external_dataset_path: Optional[str] = None,
    max_samples: int = 20000,
) -> Tuple[List[str], List[str]]:
    """
    Load external training data when a path is provided, otherwise use SAMPLE_POSTS.
    """
    if external_dataset_path:
        print(f"Using external dataset: {external_dataset_path}")
        texts, labels = load_external_dataset(external_dataset_path, max_samples)
    else:
        print("Using internal dataset")
        texts = list(SAMPLE_POSTS)
        labels = list(TRUE_LABELS)

    print(f"Dataset size: {len(texts)}")
    return texts, labels


def split_dataset(
    texts: List[str],
    labels: List[str],
    test_size: float = 0.2,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split the dataset into train and test sets.
    """
    label_counts = Counter(labels)
    test_count = max(1, int(round(len(texts) * test_size)))
    can_stratify = (
        len(label_counts) > 1
        and min(label_counts.values()) >= 2
        and test_count >= len(label_counts)
    )

    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels if can_stratify else None,
    )


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> float:
    """
    Evaluate the trained model on a labeled dataset.

    Prints each text with its predicted label and the true label,
    then returns the overall accuracy as a float between 0 and 1.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check your dataset."
        )

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    print("=== ML Model Evaluation on Dataset ===")
    correct = 0
    for text, true_label, pred_label in zip(texts, labels, preds):
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        print(
            f'"{safe_for_console(text)}" -> '
            f"predicted={pred_label}, true={true_label}"
        )

    accuracy = accuracy_score(labels, preds)
    print(f"\nAccuracy on this dataset: {accuracy:.2f}")
    return accuracy


def predict_single_text(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using
    the trained ML model.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def predict_with_confidence(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> Tuple[str, float]:
    """
    Predict the mood label and confidence for a single text string.
    """
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    best_index = probabilities.argmax()
    label = model.classes_[best_index]
    confidence = float(probabilities[best_index])

    return label, confidence


def run_interactive_loop(
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> None:
    """
    Let the user type their own sentences and see the ML model's
    predicted mood label.

    Type 'quit' or press Enter on an empty line to exit.
    """
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the ML Mood Machine.")
    parser.add_argument(
        "external_dataset",
        nargs="?",
        help="Optional path to an external sentiment CSV dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Maximum number of external rows to load.",
    )
    args = parser.parse_args()

    texts, labels = load_training_data(args.external_dataset, args.max_samples)
    train_texts, test_texts, train_labels, test_labels = split_dataset(texts, labels)

    print(f"Training size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}\n")

    vectorizer, model = train_ml_model(train_texts, train_labels)

    test_accuracy = evaluate_on_dataset(test_texts, test_labels, vectorizer, model)
    print(f"Test accuracy: {test_accuracy:.2f}")

    # Let the user try their own examples.
    run_interactive_loop(vectorizer, model)

    print("\nTip: Compare these predictions with the rule based model")
    print("by running `python main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
