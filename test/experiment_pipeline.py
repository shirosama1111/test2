import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_utils import LABELS, SEVERITY_LABELS, build_text, load_jsonl, safe_label

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


def _normalize_split(value: Any) -> str:
    return str(value).lower() if value is not None else "train"


def compute_dataset_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [safe_label(r.get("ground_truth_label"), LABELS, "UI") for r in records]
    severities = [safe_label(r.get("severity"), SEVERITY_LABELS, "MEDIUM") for r in records]
    splits = [_normalize_split(r.get("metadata", {}).get("split")) for r in records]

    label_counts = Counter(labels)
    severity_counts = Counter(severities)
    split_counts = Counter(splits)

    text_lengths = [len(build_text(r)) for r in records]
    missing_fields = {"bug_description": 0, "steps_to_reproduce": 0, "console_logs": 0, "error_context": 0}
    for rec in records:
        if not rec.get("bug_description"):
            missing_fields["bug_description"] += 1
        if not rec.get("steps_to_reproduce"):
            missing_fields["steps_to_reproduce"] += 1
        if not rec.get("console_logs"):
            missing_fields["console_logs"] += 1
        if not rec.get("error_context"):
            missing_fields["error_context"] += 1

    return {
        "n_samples": len(records),
        "label_distribution": {k: label_counts.get(k, 0) for k in LABELS},
        "severity_distribution": {k: severity_counts.get(k, 0) for k in SEVERITY_LABELS},
        "split_distribution": {k: split_counts.get(k, 0) for k in ["train", "val", "test"]},
        "avg_text_length": round(float(np.mean(text_lengths)) if text_lengths else 0.0, 2),
        "min_text_length": int(min(text_lengths)) if text_lengths else 0,
        "max_text_length": int(max(text_lengths)) if text_lengths else 0,
        "missing_fields": missing_fields,
        "duplicate_ids": len(records) - len({r.get("metadata", {}).get("id") for r in records if r.get("metadata", {}).get("id") is not None}),
    }


def compute_classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_label": {label: report.get(label, {}) for label in labels},
    }


def _predict_rule_based(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from rule_based_system import RuleBasedBugClassifier

    model = RuleBasedBugClassifier()
    return [model.predict(r) for r in records]


def _predict_ml(records: Iterable[Dict[str, Any]], train_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from ml_system import SklearnBugClassifier

    model = SklearnBugClassifier()
    model.fit(train_records)
    return [model.predict(r) for r in records]


def _build_second_opinion_prompt(record: Dict[str, Any], few_shot: List[Dict[str, Any]]) -> str:
    text = build_text(record)
    prompt = """
Bạn là người đánh giá lại báo cáo lỗi phần mềm.
Nhiệm vụ: dự đoán
- ground_truth_label: UI, API, LOGIC
- severity: LOW, MEDIUM, HIGH, CRITICAL
Chỉ trả về JSON hợp lệ, không giải thích thêm.

Ví dụ định dạng:
{"ground_truth_label": "UI", "severity": "MEDIUM"}

Báo cáo lỗi:
"""" + text + """
"""
    if few_shot:
        prompt += "\nVí dụ huấn luyện:\n"
        for ex in few_shot[:3]:
            prompt += json.dumps({
                "input": build_text(ex),
                "output": {
                    "ground_truth_label": safe_label(ex.get("ground_truth_label"), LABELS, "UI"),
                    "severity": safe_label(ex.get("severity"), SEVERITY_LABELS, "MEDIUM"),
                },
            }, ensure_ascii=False) + "\n"
    return prompt


def _predict_second_opinion(records: List[Dict[str, Any]], train_records: List[Dict[str, Any]], model_name: str, api_key: str | None) -> List[Dict[str, Any]]:
    if OpenAI is None or not api_key:
        raise RuntimeError("openai package or OPENAI_API_KEY is not available")

    client = OpenAI(api_key=api_key)
    few_shot = train_records[:5]
    preds: List[Dict[str, Any]] = []
    for rec in records:
        prompt = _build_second_opinion_prompt(rec, few_shot)
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Trả về đúng JSON không có markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            label = safe_label(data.get("ground_truth_label"), LABELS, "UI")
            sev = safe_label(data.get("severity"), SEVERITY_LABELS, "MEDIUM")
            preds.append({"ground_truth_label": label, "severity": sev})
        except Exception:
            preds.append({"ground_truth_label": "UI", "severity": "MEDIUM"})
    return preds


def _collect_test_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train = [r for r in records if _normalize_split(r.get("metadata", {}).get("split")) == "train"]
    test = [r for r in records if _normalize_split(r.get("metadata", {}).get("split")) == "test"]
    if not test and records:
        test = records[: min(100, len(records))]
    if not train and records:
        train = records[: max(1, min(50, len(records)))]
    return train, test


def _evaluate_predictions(test_records: List[Dict[str, Any]], predictions: List[Dict[str, Any]], label_field: str = "ground_truth_label") -> Dict[str, Any]:
    y_true_type = [safe_label(r.get(label_field), LABELS, "UI") for r in test_records]
    y_pred_type = [safe_label(p.get(label_field), LABELS, "UI") for p in predictions]

    y_true_sev = [safe_label(r.get("severity"), SEVERITY_LABELS, "MEDIUM") for r in test_records]
    y_pred_sev = [safe_label(p.get("severity"), SEVERITY_LABELS, "MEDIUM") for p in predictions]

    return {
        "type": compute_classification_metrics(y_true_type, y_pred_type, LABELS),
        "severity": compute_classification_metrics(y_true_sev, y_pred_sev, SEVERITY_LABELS),
    }


def run_experiment(records: List[Dict[str, Any]], system_name: str, model_name: str = "gpt-4o-mini", api_key: str | None = None, repeats: int = 1, seed: int = 42) -> Dict[str, Any]:
    train_records, test_records = _collect_test_records(records)
    if not test_records:
        raise ValueError("No test records available for evaluation.")

    random.seed(seed)
    rng = random.Random(seed)

    summaries: List[Dict[str, Any]] = []
    for run_idx in range(repeats):
        run_records = test_records
        if repeats > 1:
            run_records = rng.sample(test_records, k=min(len(test_records), max(10, len(test_records) // 2)))

        if system_name == "rule":
            preds = _predict_rule_based(run_records)
        elif system_name == "ml":
            preds = _predict_ml(run_records, train_records)
        elif system_name == "llm":
            if OpenAI is None or not api_key:
                preds = [{"ground_truth_label": "UI", "severity": "MEDIUM"} for _ in run_records]
            else:
                preds = _predict_second_opinion(run_records, train_records, model_name, api_key)
        elif system_name == "second_opinion":
            preds = _predict_second_opinion(run_records, train_records, model_name, api_key)
        else:
            raise ValueError(f"Unsupported system: {system_name}")

        summary = _evaluate_predictions(run_records, preds)
        summary["run_index"] = run_idx
        summary["n_eval_samples"] = len(run_records)
        summaries.append(summary)

    avg_type = {
        key: float(np.mean([s["type"][key] for s in summaries]))
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    }
    avg_sev = {
        key: float(np.mean([s["severity"][key] for s in summaries]))
        for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    }

    return {
        "system": system_name,
        "repeats": repeats,
        "runs": summaries,
        "average_type": avg_type,
        "average_severity": avg_sev,
    }


def run_second_opinion_agreement(records: List[Dict[str, Any]], model_name: str = "gpt-4o-mini", api_key: str | None = None) -> Dict[str, Any]:
    train_records, test_records = _collect_test_records(records)
    if not test_records:
        raise ValueError("No test records available for second-opinion evaluation.")

    if OpenAI is None or not api_key:
        return {"status": "skipped", "reason": "openai package or OPENAI_API_KEY is not available"}

    predictions = _predict_second_opinion(test_records, train_records, model_name, api_key)
    y_true_type = [safe_label(r.get("ground_truth_label"), LABELS, "UI") for r in test_records]
    y_pred_type = [safe_label(p.get("ground_truth_label"), LABELS, "UI") for p in predictions]
    agreement = sum(1 for a, b in zip(y_true_type, y_pred_type) if a == b) / max(len(y_true_type), 1)
    return {
        "status": "ok",
        "agreement_rate": float(agreement),
        "n_samples": len(test_records),
        "type_metrics": compute_classification_metrics(y_true_type, y_pred_type, LABELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset assessment and repeatable experiment pipeline for bug reports.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL file")
    parser.add_argument("--scenario", choices=["rule", "ml", "llm", "second_opinion", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=3, help="How many repeated runs per system")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name for second-opinion evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit to evaluate only the first N rows")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = load_jsonl(str(dataset_path))
    if args.limit is not None:
        records = records[: args.limit]

    stats = compute_dataset_statistics(records)
    print("=== Dataset statistics ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    api_key = os.getenv("OPENAI_API_KEY")
    systems = [args.scenario] if args.scenario != "all" else ["rule", "ml", "llm", "second_opinion"]
    results = {
        "dataset_path": str(dataset_path),
        "scenario": args.scenario,
        "dataset_statistics": stats,
        "experiments": [],
    }

    for system_name in systems:
        try:
            if system_name == "second_opinion":
                result = run_second_opinion_agreement(records, args.model, api_key)
            else:
                result = run_experiment(records, system_name, args.model, api_key, repeats=args.repeats, seed=args.seed)
            results["experiments"].append(result)
            print(f"\n=== Result: {system_name} ===")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:  # pragma: no cover
            print(f"\n=== Result: {system_name} FAILED ===")
            print(str(exc))
            results["experiments"].append({"system": system_name, "status": "failed", "reason": str(exc)})

    output_path = dataset_path.with_suffix(".experiment_summary.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
