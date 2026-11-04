import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from dataset_utils import (
    build_text,
    load_jsonl,
    safe_label,
    LABELS,
    SEVERITY_LABELS
)


class SklearnBugClassifier:
    def __init__(self):
        self.type_model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            ),
            ("clf", LogisticRegression(max_iter=2000))
        ])

        self.severity_model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            ),
            ("clf", LogisticRegression(max_iter=2000))
        ])

    def fit(self, train_records):
        X_type = [build_text(r) for r in train_records]
        y_type = [
            safe_label(r.get("ground_truth_label"), LABELS, "UI")
            for r in train_records
        ]

        X_sev = [build_text(r) for r in train_records]
        y_sev = [
            safe_label(r.get("severity"), SEVERITY_LABELS, "MEDIUM")
            for r in train_records
        ]

        self.type_model.fit(X_type, y_type)
        self.severity_model.fit(X_sev, y_sev)

    def predict(self, record):
        text = build_text(record)

        label = self.type_model.predict([text])[0]
        severity = self.severity_model.predict([text])[0]

        return {
            "ground_truth_label": label,
            "severity": severity
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ml_system.py path/to/dataset.jsonl")
        sys.exit(1)

    records = load_jsonl(sys.argv[1])

    train = [
        r for r in records
        if str(r.get("metadata", {}).get("split", "train")).lower()
        == "train"
    ]

    test = [
        r for r in records
        if str(r.get("metadata", {}).get("split", "train")).lower()
        == "test"
    ]

    print(f"Train: {len(train)} | Test: {len(test)}")

    if not train:
        print("Loi: Tap Train rong!")
        sys.exit(1)

    if not test:
        print("Loi: Tap Test rong, khong the danh gia!")
        sys.exit(1)

    model = SklearnBugClassifier()
    model.fit(train)

    preds = [model.predict(r) for r in test]

    y_true_type = [
        safe_label(r.get("ground_truth_label"), LABELS, "UI")
        for r in test
    ]

    y_true_severity = [
        safe_label(r.get("severity"), SEVERITY_LABELS, "MEDIUM")
        for r in test
    ]

    y_pred_type = [p["ground_truth_label"] for p in preds]
    y_pred_severity = [p["severity"] for p in preds]

    print("\n=== Bug Type Classification ===")

    print("Accuracy:", accuracy_score(y_true_type, y_pred_type))

    print(classification_report(
        y_true_type,
        y_pred_type,
        labels=LABELS,
        zero_division=0
    ))

    print("\n=== Severity Classification ===")

    print("Accuracy:", accuracy_score(
        y_true_severity,
        y_pred_severity
    ))

    print(classification_report(
        y_true_severity,
        y_pred_severity,
        labels=SEVERITY_LABELS,
        zero_division=0
    ))

    print("\n=== Sample Predictions ===")

    print(json.dumps(
        preds[:5],
        ensure_ascii=False,
        indent=2
    ))
