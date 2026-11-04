import json
import re
from typing import List, Dict, Any

LABELS = ["UI", "API", "LOGIC"]
SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join([str(x) for x in value])
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def get_split(record: Dict[str, Any]) -> str:
    meta = record.get("metadata", {})
    if isinstance(meta, dict):
        return str(meta.get("split", "train")).lower()
    return "train"

def build_text(record: Dict[str, Any]) -> str:
    text_parts = []
    text_parts.append(record.get("bug_description", ""))
    steps = record.get("steps_to_reproduce", [])
    if isinstance(steps, list):
        text_parts.append(" ".join([str(s) for s in steps]))
    logs = record.get("console_logs", [])
    if isinstance(logs, list):
        text_parts.append(" ".join([str(l) for l in logs]))
    context = record.get("error_context", {})
    if isinstance(context, dict):
        text_parts.append(normalize_text(context))
    return normalize_text(" ".join(text_parts))

def safe_label(value: Any, choices: List[str], default: str) -> str:
    if isinstance(value, str):
        v = value.strip().upper()
        if v in choices:
            return v
    return default