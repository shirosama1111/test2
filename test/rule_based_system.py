import json
import re
import sys
from dataset_utils import build_text, load_jsonl
from sklearn.metrics import accuracy_score, classification_report
from dataset_utils import safe_label, LABELS, SEVERITY_LABELS

class RuleBasedBugClassifier:
    def __init__(self):
        self.ui_keywords = {
            "button": 3,
            "screen": 3,
            "layout": 4,
            "render": 4,
            "display": 2,
            "blank page": 5,
            "misalign": 5,
            "not visible": 5,
            "visible": 2,
            "dropdown": 3,
            "modal": 3,
            "checkbox": 3,
            "input field": 2,
            "form": 2,
            "click": 2,
            "scroll": 2,
            "overflow": 4,
        }

        self.api_keywords = {
            "api": 4,
            "fetch": 4,
            "axios": 4,
            "request": 3,
            "response": 3,
            "status code": 5,
            "500": 6,
            "401": 5,
            "403": 5,
            "timeout": 5,
            "cors": 5,
            "endpoint": 4,
            "network": 3,
            "server error": 5,
            "json": 2,
            "backend": 3,
        }

        self.logic_keywords = {
            "wrong value": 6,
            "incorrect": 4,
            "calculation": 5,
            "sum": 5,
            "total": 5,
            "discount": 5,
            "price": 4,
            "tax": 4,
            "balance": 4,
            "validation": 4,
            "condition": 3,
            "logic": 4,
            "expected": 3,
            "actual": 3,
            "not updating": 5,
            "inconsistent": 4,
            "filter": 3,
            "sort": 3,
            "count mismatch": 5,
            "wrong result": 5,
        }

    def _score_keywords(self, text: str, mapping: dict) -> int:
        score = 0
        for keyword, weight in mapping.items():
            if keyword in text:
                score += weight
        return score

    def predict_type(self, text: str) -> str:
        t = text.lower()

        scores = {
            "UI": self._score_keywords(t, self.ui_keywords),
            "API": self._score_keywords(t, self.api_keywords),
            "LOGIC": self._score_keywords(t, self.logic_keywords),
        }

        if re.search(r"\b(click|tap|hover|scroll|open|close|submit)\b.*\b(button|menu|modal|page|screen|dropdown|checkbox|field)\b", t):
            scores["UI"] += 6

        if re.search(r"\b(status code|http status|response|request|fetch|axios|endpoint|api)\b", t):
            scores["API"] += 4

        if re.search(r"\b(expected|should|but got|actual)\b.*\b(total|sum|price|discount|tax|count|calculation|balance)\b", t):
            scores["LOGIC"] += 7

        if re.search(r"\b(button|layout|screen|render|visible|display|blank)\b.*\b(not|broken|misalign|missing|overlap|overflow)\b", t):
            scores["UI"] += 8

        if re.search(r"\b(total|sum|price|discount|count|tax|balance)\b.*\b(wrong|incorrect|mismatch|not match)\b", t):
            scores["LOGIC"] += 8

        if re.search(r"\b(status code\s*500|500\s*error|401|403|timeout|cors|network error)\b", t):
            scores["API"] += 8

        best = max(scores, key=scores.get)
        if max(scores.values()) <= 0:
            return "UI"

        if scores["UI"] >= 7 and scores["API"] <= scores["UI"]:
            return "UI"
        if scores["LOGIC"] >= 7 and scores["API"] <= scores["LOGIC"]:
            return "LOGIC"

        return best

    def predict_severity(self, text: str, label: str) -> str:
        t = text.lower()

        critical_signals = [
            "data loss", "security", "fatal", "crash", "cannot login",
            "payment failed", "server down", "all users affected", "critical bug"
        ]
        high_signals = [
            "500", "503", "401", "403", "timeout", "api failed", "network error",
            "wrong result", "inconsistent data", "payment issue", "cannot save"
        ]
        medium_signals = [
            "incorrect total", "wrong value", "not updating", "button not working",
            "login issue", "logic mismatch", "blank screen", "slow", "delay"
        ]
        low_signals = [
            "typo", "minor alignment", "color mismatch", "small cosmetic",
            "label wrong", "text overlap"
        ]

        if any(s in t for s in critical_signals):
            return "CRITICAL"
        if any(s in t for s in high_signals):
            return "HIGH"
        if any(s in t for s in medium_signals):
            return "MEDIUM"
        if any(s in t for s in low_signals):
            return "LOW"

        # fallback theo label
        if label == "API":
            return "MEDIUM"
        if label == "LOGIC":
            return "MEDIUM"
        if label == "UI":
            return "LOW"

        return "MEDIUM"

    def predict(self, record):
        text = build_text(record)
        label = self.predict_type(text)
        severity = self.predict_severity(text, label)
        return {
            "ground_truth_label": label,
            "severity": severity
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rule_based_system.py path/to/dataset.jsonl")
        sys.exit(1)

    records = load_jsonl(sys.argv[1])
    test = [
        r for r in records
        if str(r.get("metadata", {}).get("split", "train")).lower() == "test"
    ]
    
    model = RuleBasedBugClassifier()
    preds = [model.predict(r) for r in test]

    print("\n=== Sample Predictions ===")
    print(json.dumps(preds[:10], ensure_ascii=False, indent=2))

    if test:
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
        print("Accuracy:", accuracy_score(y_true_severity, y_pred_severity))
        print(classification_report(
            y_true_severity,
            y_pred_severity,
            labels=SEVERITY_LABELS,
            zero_division=0
        ))
    else:
        print("Khong co du lieu Test de danh gia.")

    print(f"Total test samples: {len(test)}")