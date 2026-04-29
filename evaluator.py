"""
Compare the rule-based and ML Mood Machine models with confidence scores.
"""

import argparse

from data_loader import load_external_dataset
from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer
from ml_experiments import (
    safe_for_console,
    split_dataset,
    train_ml_model,
    predict_with_confidence,
)


def summarize_results(name, predictions, labels, confidences):
    """
    Print accuracy and average confidence for one model.
    """
    total = len(labels)
    correct = sum(pred == true for pred, true in zip(predictions, labels))
    accuracy = correct / total if total else 0.0
    avg_confidence = sum(confidences) / total if total else 0.0

    print(f"{name} accuracy: {accuracy:.2f}")
    print(f"{name} avg confidence: {avg_confidence:.2f}")


def evaluate_rule_based():
    """
    Run the rule-based analyzer on the shared dataset.
    """
    analyzer = MoodAnalyzer()
    predictions = []
    confidences = []

    print("=== Rule-based Evaluation ===")
    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        predicted_label, confidence = analyzer.predict_label(
            text,
            return_confidence=True,
        )
        predictions.append(predicted_label)
        confidences.append(confidence)

        print(f"Text: {safe_for_console(text)}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}\n")

    summarize_results("Rule-based", predictions, TRUE_LABELS, confidences)


def evaluate_ml():
    """
    Train and run the ML analyzer on the shared dataset.
    """
    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
    predictions = []
    confidences = []

    print("\n=== ML Evaluation ===")
    for text, true_label in zip(SAMPLE_POSTS, TRUE_LABELS):
        predicted_label, confidence = predict_with_confidence(text, vectorizer, model)
        predictions.append(predicted_label)
        confidences.append(confidence)

        print(f"Text: {safe_for_console(text)}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}\n")

    summarize_results("ML", predictions, TRUE_LABELS, confidences)


def evaluate_external_ml(external_dataset_path, max_samples=20000):
    """
    Train and evaluate the ML analyzer on an optional external dataset.
    """
    print(f"\nUsing external dataset: {external_dataset_path}")
    texts, labels = load_external_dataset(external_dataset_path, max_samples)
    print(f"Dataset size: {len(texts)}")

    train_texts, test_texts, train_labels, test_labels = split_dataset(texts, labels)
    print(f"Training size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")

    vectorizer, model = train_ml_model(train_texts, train_labels)
    predictions = []
    confidences = []

    print("\n=== External ML Evaluation ===")
    for text, true_label in zip(test_texts, test_labels):
        predicted_label, confidence = predict_with_confidence(text, vectorizer, model)
        predictions.append(predicted_label)
        confidences.append(confidence)

        print(f"Text: {safe_for_console(text)}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}\n")

    summarize_results("External ML", predictions, test_labels, confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mood Machine reliability.")
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

    evaluate_rule_based()
    evaluate_ml()

    if args.external_dataset:
        evaluate_external_ml(args.external_dataset, args.max_samples)
