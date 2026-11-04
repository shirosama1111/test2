import subprocess
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python run_all.py path/to/dataset.jsonl")
    sys.exit(1)

dataset_path = sys.argv[1]

if not os.path.isfile(dataset_path):
    print(f"Error: Dataset not found: {dataset_path}")
    sys.exit(1)

systems = [
    ("Rule-based system", "rule_based_system.py"),
    ("ML system", "ml_system.py"),
    ("LLM system", "llm_system.py"),
]

for name, script in systems:
    print(f"\n{'=' * 50}")
    print(f"=== {name} ===")
    print(f"{'=' * 50}")

    if not os.path.isfile(script):
        print(f"Error: Cannot find {script}")
        continue

    try:
        result = subprocess.run(
            [sys.executable, script, dataset_path],
            check=False
        )

        if result.returncode != 0:
            print(f"Warning: {script} exited with code {result.returncode}")

    except Exception as e:
        print(f"Error running {script}: {e}")

print("\n=== All systems finished ===")