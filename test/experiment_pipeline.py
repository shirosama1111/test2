"""Dataset assessment and repeatable experiments for bug-report classification.

The second-opinion LLM sees only report text, never dataset labels or split examples.
Its agreement with synthetic/unverified labels is a diagnostic, not proof of truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_utils import LABELS, SEVERITY_LABELS, build_text, load_jsonl, safe_label

try:
    from openai import OpenAI
except ImportError:  # OpenAI SDK is only needed for --scenario second-opinion/all.
    OpenAI = None

SEVERITY_RANK = {label: rank for rank, label in enumerate(SEVERITY_LABELS)}


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def _valid_label(value: Any, choices: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().upper()
    return label if label in choices else None


def _field_present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def assess_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return reproducible descriptive statistics and basic data-quality checks."""
    labels = Counter(str(r.get("ground_truth_label", "<missing>")) for r in records)
    severities = Counter(str(r.get("severity", "<missing>")) for r in records)
    splits = Counter(str(_metadata(r).get("split", "train")).lower() for r in records)
    text_lengths = [len(build_text(r)) for r in records]
    ids = [str(_metadata(r)["id"]) for r in records if _metadata(r).get("id") is not None]
    texts = [build_text(r).casefold() for r in records if build_text(r)]
    annotation_status = Counter(str(_metadata(r).get("annotation_status", "<missing>")) for r in records)
    missing = {
        field: sum(not _field_present(record.get(field)) for record in records)
        for field in ("bug_description", "steps_to_reproduce", "console_logs", "error_context", "root_cause", "resolution")
    }
    invalid_labels = sum(_valid_label(r.get("ground_truth_label"), LABELS) is None for r in records)
    invalid_severity = sum(_valid_label(r.get("severity"), SEVERITY_LABELS) is None for r in records)

    return {
        "n_samples": len(records),
        "label_distribution": dict(sorted(labels.items())),
        "severity_distribution": dict(sorted(severities.items())),
        "split_distribution": dict(sorted(splits.items())),
        "annotation_status_distribution": dict(sorted(annotation_status.items())),
        "text_length_chars": {
            "mean": round(statistics.mean(text_lengths), 2) if text_lengths else 0,
            "median": statistics.median(text_lengths) if text_lengths else 0,
            "min": min(text_lengths, default=0),
            "max": max(text_lengths, default=0),
        },
        "missing_or_empty_fields": missing,
        "duplicate_ids": len(ids) - len(set(ids)),
        "duplicate_nonempty_texts": len(texts) - len(set(texts)),
        "invalid_type_labels": invalid_labels,
        "invalid_severity_labels": invalid_severity,
        "warnings": [
            "Nếu annotation_status là synthetic-unverified, các metric phản ánh mức khớp với nhãn hiện có, không khẳng định độ đúng thực tế.",
            "Kiểm tra rò rỉ giữa train/test theo kịch bản gốc hoặc template trước khi diễn giải kết quả.",
            "Thống kê thiếu trường không đồng nghĩa lỗi dữ liệu: một số trường có thể tùy chọn.",
        ],
    }


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    if not y_true:
        raise ValueError("Không có mẫu hợp lệ để tính metric.")
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_label": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def evaluate_predictions(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError("Số prediction không khớp số record.")
    type_pairs = [
        (_valid_label(r.get("ground_truth_label"), LABELS), _valid_label(p.get("ground_truth_label"), LABELS))
        for r, p in zip(records, predictions)
    ]
    severity_pairs = [
        (_valid_label(r.get("severity"), SEVERITY_LABELS), _valid_label(p.get("severity"), SEVERITY_LABELS))
        for r, p in zip(records, predictions)
    ]
    type_pairs = [(a, b) for a, b in type_pairs if a is not None and b is not None]
    severity_pairs = [(a, b) for a, b in severity_pairs if a is not None and b is not None]
    if not type_pairs or not severity_pairs:
        raise ValueError("Không đủ nhãn hợp lệ để đánh giá.")
    y_type, p_type = map(list, zip(*type_pairs))
    y_sev, p_sev = map(list, zip(*severity_pairs))
    ordinal_mae = statistics.mean(abs(SEVERITY_RANK[a] - SEVERITY_RANK[b]) for a, b in severity_pairs)
    return {
        "bug_type": classification_metrics(y_type, p_type, LABELS),
        "severity": classification_metrics(y_sev, p_sev, SEVERITY_LABELS),
        "severity_ordinal_mae": float(ordinal_mae),
        "invalid_or_missing_gold": {"bug_type": len(records) - len(type_pairs), "severity": len(records) - len(severity_pairs)},
    }


def _predict(system: str, train: list[dict[str, Any]], test: list[dict[str, Any]], seed: int,
             model_name: str, llm_limit: int | None) -> list[dict[str, Any]]:
    if system == "rule":
        from rule_based_system import RuleBasedBugClassifier
        classifier = RuleBasedBugClassifier()
        return [classifier.predict(record) for record in test]
    if system == "ml":
        from ml_system import SklearnBugClassifier
        random.seed(seed)
        classifier = SklearnBugClassifier()
        classifier.fit(train)
        return [classifier.predict(record) for record in test]
    if system != "second-opinion":
        raise ValueError(f"Scenario không hỗ trợ: {system}")
    if OpenAI is None:
        raise RuntimeError("Cần cài OpenAI SDK: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu biến môi trường OPENAI_API_KEY")

    selected = test if llm_limit is None else test[:llm_limit]
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        # Do not expose gold labels, root causes, resolutions, or training examples.
        prompt = (
            "Phân loại báo cáo lỗi phần mềm. Chọn bug_type trong UI, API, LOGIC; "
            "severity trong LOW, MEDIUM, HIGH, CRITICAL. Dựa trên nội dung báo cáo; "
            "nếu thiếu bằng chứng, chọn nhãn hợp lý nhất. Chỉ trả JSON với các khóa "
            "bug_type và severity.\n\nBáo cáo:\n" + build_text(record)
        )
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            data = json.loads(response.choices[0].message.content or "{}")
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"LLM trả JSON không hợp lệ ở mẫu {index}") from exc
        bug_type = _valid_label(data.get("bug_type", data.get("ground_truth_label")), LABELS)
        severity = _valid_label(data.get("severity"), SEVERITY_LABELS)
        if bug_type is None or severity is None:
            raise RuntimeError(f"LLM trả nhãn ngoài schema ở mẫu {index}: {data!r}")
        predictions.append({"ground_truth_label": bug_type, "severity": severity})
        if index % 20 == 0:
            print(f"LLM đã đánh giá {index}/{len(selected)} mẫu")
    if len(selected) != len(test):
        print(f"Cảnh báo: LLM đánh giá {len(selected)}/{len(test)} mẫu test theo --llm-limit.")
    return predictions


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = [
        ("bug_type", "accuracy"), ("bug_type", "macro_f1"), ("bug_type", "weighted_f1"),
        ("severity", "accuracy"), ("severity", "macro_f1"), ("severity", "weighted_f1"),
        (None, "severity_ordinal_mae"),
    ]
    summary: dict[str, Any] = {}
    for section, metric in metric_paths:
        values = [run["metrics"][section][metric] if section else run["metrics"][metric] for run in runs]
        key = f"{section}.{metric}" if section else metric
        summary[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    return summary


def _write_outputs(output_dir: Path, results: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "run", "seed", "n_test", "type_accuracy", "type_macro_f1", "type_weighted_f1", "severity_accuracy", "severity_macro_f1", "severity_weighted_f1", "severity_ordinal_mae"])
        writer.writeheader()
        for experiment in results["experiments"]:
            for run in experiment.get("runs", []):
                metrics = run["metrics"]
                writer.writerow({
                    "system": experiment["system"], "run": run["run"], "seed": run["seed"], "n_test": run["n_test"],
                    "type_accuracy": metrics["bug_type"]["accuracy"], "type_macro_f1": metrics["bug_type"]["macro_f1"],
                    "type_weighted_f1": metrics["bug_type"]["weighted_f1"], "severity_accuracy": metrics["severity"]["accuracy"],
                    "severity_macro_f1": metrics["severity"]["macro_f1"], "severity_weighted_f1": metrics["severity"]["weighted_f1"],
                    "severity_ordinal_mae": metrics["severity_ordinal_mae"],
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Thống kê dataset và chạy thí nghiệm phân loại bug có thể lặp lại.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "bug_dataset.jsonl"), help="Đường dẫn JSONL")
    parser.add_argument("--scenario", choices=["stats", "rule", "ml", "second-opinion", "all"], default="all")
    parser.add_argument("--repeats", type=int, default=1, help="Số lần chạy; LLM sẽ gọi lại và có thể phát sinh chi phí")
    parser.add_argument("--seed", type=int, default=42, help="Seed gốc, mỗi lần chạy dùng seed + run index")
    parser.add_argument("--model", default=os.getenv("SECOND_OPINION_MODEL", "gpt-4o-mini"), help="Model LLM đánh giá độc lập")
    parser.add_argument("--llm-limit", type=int, default=None, help="Chỉ gọi LLM trên N mẫu đầu của test; mặc định dùng toàn bộ test")
    parser.add_argument("--output-dir", default=None, help="Thư mục output; mặc định test/runs/<UTC timestamp>")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats phải >= 1")
    if args.llm_limit is not None and args.llm_limit < 1:
        parser.error("--llm-limit phải >= 1")

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")
    records = load_jsonl(str(dataset_path))
    if not records:
        raise ValueError("Dataset rỗng")
    stats = assess_dataset(records)
    print("=== Dataset assessment ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.scenario == "stats":
        return

    train = [r for r in records if str(_metadata(r).get("split", "train")).lower() == "train"]
    test = [r for r in records if str(_metadata(r).get("split", "train")).lower() == "test"]
    if not train or not test:
        raise ValueError("Cần có cả split train và test; pipeline không tự tạo split để tránh đánh giá sai.")

    systems = ["rule", "ml", "second-opinion"] if args.scenario == "all" else [args.scenario]
    experiments = []
    for system in systems:
        print(f"\n=== {system} ===")
        run_results = []
        for run_index in range(args.repeats):
            run_seed = args.seed + run_index
            started = time.perf_counter()
            predictions = _predict(system, train, test, run_seed, args.model, args.llm_limit)
            evaluated_records = test[:len(predictions)]
            metrics = evaluate_predictions(evaluated_records, predictions)
            run_results.append({"run": run_index + 1, "seed": run_seed, "n_test": len(evaluated_records), "elapsed_seconds": round(time.perf_counter() - started, 3), "metrics": metrics})
            print(f"Run {run_index + 1}/{args.repeats}: type macro-F1={metrics['bug_type']['macro_f1']:.4f}, severity macro-F1={metrics['severity']['macro_f1']:.4f}")
        experiments.append({"system": system, "model": args.model if system == "second-opinion" else None, "runs": run_results, "aggregate": _summarize_runs(run_results)})

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROOT / "runs" / run_id
    results = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "scenario": args.scenario,
        "repeats": args.repeats,
        "base_seed": args.seed,
        "dataset_assessment": stats,
        "experiments": experiments,
        "interpretation_note": "LLM agreement compares model outputs with existing annotations; synthetic/unverified labels require human review.",
    }
    _write_outputs(output_dir, results)
    print(f"\nĐã lưu summary.json và metrics.csv tại {output_dir}")


if __name__ == "__main__":
    main()
