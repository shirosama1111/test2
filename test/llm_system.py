import json
import os
import sys

from dataset_utils import (
    build_text,
    load_jsonl,
    safe_label,
    LABELS,
    SEVERITY_LABELS
)

from sklearn.metrics import accuracy_score, classification_report

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMBugClassifier:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("Bạn cần cài openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def build_prompt(self, record, few_shot=None):
        text = build_text(record)

        prompt = f"""
Bạn là hệ thống phân loại báo cáo lỗi phần mềm.

Nhiệm vụ:
- Chọn label: UI, API, LOGIC
- Chọn severity: LOW, MEDIUM, HIGH, CRITICAL
- Chỉ trả về JSON hợp lệ, không giải thích.

Ví dụ schema:
{{"ground_truth_label":"UI","severity":"MEDIUM"}}

Báo cáo lỗi:
{text}
"""

        if few_shot:
            prompt += "\nVí dụ học tập:\n"

            for ex in few_shot[:3]:
                prompt += json.dumps({
                    "input": build_text(ex),
                    "output": {
                        "ground_truth_label": safe_label(
                            ex.get("ground_truth_label"),
                            LABELS,
                            "UI"
                        ),
                        "severity": safe_label(
                            ex.get("severity"),
                            SEVERITY_LABELS,
                            "MEDIUM"
                        )
                    }
                }, ensure_ascii=False) + "\n"

        return prompt

    def predict(self, record, few_shot=None):
        prompt = self.build_prompt(record, few_shot=few_shot)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Trả về đúng JSON không có markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.choices[0].message.content

        try:
            data = json.loads(content)

            label = safe_label(
                data.get("ground_truth_label"),
                LABELS,
                "UI"
            )

            severity = safe_label(
                data.get("severity"),
                SEVERITY_LABELS,
                "MEDIUM"
            )

            return {
                "ground_truth_label": label,
                "severity": severity
            }

        except (json.JSONDecodeError, TypeError):
            return {
                "ground_truth_label": "UI",
                "severity": "MEDIUM"
            }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python llm_system.py path/to/dataset.jsonl")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "Thiếu OPENAI_API_KEY. "
            "Hãy thiết lập biến môi trường trước khi chạy."
        )

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
        print("Loi: Tap Test rong!")
        sys.exit(1)

    few_shot = train[:5]

    model = LLMBugClassifier(
        api_key=api_key,
        model="gpt-4o-mini"
    )

    eval_records = test[:10]

    preds = [
        model.predict(r, few_shot=few_shot)
        for r in eval_records
    ]

    y_true_type = [
        safe_label(r.get("ground_truth_label"), LABELS, "UI")
        for r in eval_records
    ]

    y_true_severity = [
        safe_label(r.get("severity"), SEVERITY_LABELS, "MEDIUM")
        for r in eval_records
    ]

    y_pred_type = [
        p["ground_truth_label"]
        for p in preds
    ]

    y_pred_severity = [
        p["severity"]
        for p in preds
    ]

    print("\n=== Sample Predictions ===")

    print(json.dumps(
        preds,
        ensure_ascii=False,
        indent=2
    ))

    print("\n=== Bug Type Classification ===")

    print(
        "Accuracy:",
        accuracy_score(y_true_type, y_pred_type)
    )

    print(classification_report(
        y_true_type,
        y_pred_type,
        labels=LABELS,
        zero_division=0
    ))

    print("\n=== Severity Classification ===")

    print(
        "Accuracy:",
        accuracy_score(y_true_severity, y_pred_severity)
    )

    print(classification_report(
        y_true_severity,
        y_pred_severity,
        labels=SEVERITY_LABELS,
        zero_division=0
    ))

    print(
        f"\nEvaluated samples: {len(eval_records)}"
    )
