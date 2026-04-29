"""
Compare the rule-based and ML Mood Machine models with confidence scores.
"""

from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer
from ml_experiments import train_ml_model, predict_with_confidence


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

        print(f"Text: {text}")
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

        print(f"Text: {text}")
        print(f"True label: {true_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}\n")

    summarize_results("ML", predictions, TRUE_LABELS, confidences)


if __name__ == "__main__":
    evaluate_rule_based()
    evaluate_ml()
