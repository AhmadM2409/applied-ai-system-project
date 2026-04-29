"""
Utilities for loading optional external sentiment datasets.
"""

import csv
from typing import List, Optional, Tuple


TEXT_COLUMN_CANDIDATES = [
    "text",
    "content",
    "post",
    "tweet",
    "message",
    "sentence",
    "review",
    "comment",
]

LABEL_COLUMN_CANDIDATES = [
    "sentiment",
    "label",
    "target",
    "class",
    "category",
]


def _normalize_column_name(name: str) -> str:
    return "".join(char.lower() for char in name if char.isalnum())


def _find_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    normalized_fields = {
        _normalize_column_name(fieldname): fieldname for fieldname in fieldnames
    }

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_fields:
            return normalized_fields[normalized_candidate]

    for normalized_field, original_field in normalized_fields.items():
        if any(candidate in normalized_field for candidate in candidates):
            return original_field

    return None


def _map_label(raw_label: str) -> Optional[str]:
    label = raw_label.strip().lower()

    if "positive" in label or label in {"pos", "4"}:
        return "positive"
    if "negative" in label or label in {"neg", "0"}:
        return "negative"
    if "neutral" in label or label in {"neu", "2"}:
        return "neutral"

    return None


def load_external_dataset(path, max_samples=20000) -> Tuple[List[str], List[str]]:
    """
    Load an external sentiment CSV and map labels to project labels.

    Rows with missing text, missing label, or unsupported labels are skipped.
    """
    texts: List[str] = []
    labels: List[str] = []

    with open(path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("CSV file does not contain a header row.")

        text_column = _find_column(reader.fieldnames, TEXT_COLUMN_CANDIDATES)
        label_column = _find_column(reader.fieldnames, LABEL_COLUMN_CANDIDATES)

        if text_column is None:
            raise ValueError("Could not identify a text column in the CSV file.")
        if label_column is None:
            raise ValueError("Could not identify a label/sentiment column in the CSV file.")

        for row in reader:
            raw_text = row.get(text_column, "")
            raw_label = row.get(label_column, "")

            if raw_text is None or raw_label is None:
                continue

            text = raw_text.strip()
            label = _map_label(raw_label)

            if not text or label is None:
                continue

            texts.append(text)
            labels.append(label)

            if len(texts) >= max_samples:
                break

    return texts, labels
