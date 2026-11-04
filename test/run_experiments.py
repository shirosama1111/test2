"""Convenience CLI for the dataset experiment pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "experiment_pipeline.py"
DEFAULT_DATASET = ROOT / "data" / "bug_dataset.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick runner for dataset statistics and rule/ML/LLM comparison."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Input JSONL dataset")
    parser.add_argument(
        "--scenario",
        choices=["stats", "rule", "ml", "second-opinion", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="LLM model for second-opinion")
    parser.add_argument("--llm-limit", type=int, default=None, help="Maximum test records sent to LLM")
    parser.add_argument("--output-dir", default=None, help="Output directory for this run")
    parser.add_argument("--excel", action="store_true", help="Also write an .xlsx report (requires openpyxl)")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.llm_limit is not None and args.llm_limit < 1:
        parser.error("--llm-limit must be at least 1")
    if not PIPELINE.is_file():
        parser.error(f"Pipeline script not found: {PIPELINE}")

    command = [
        sys.executable,
        str(PIPELINE),
        "--dataset", str(Path(args.dataset).expanduser()),
        "--scenario", args.scenario,
        "--repeats", str(args.repeats),
        "--seed", str(args.seed),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.llm_limit is not None:
        command.extend(["--llm-limit", str(args.llm_limit)])
    if args.output_dir:
        command.extend(["--output-dir", args.output_dir])
    if args.excel:
        command.append("--excel")

    print("Executing:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT.parent, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
