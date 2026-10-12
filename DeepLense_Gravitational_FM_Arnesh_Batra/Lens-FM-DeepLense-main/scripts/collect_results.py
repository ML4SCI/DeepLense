"""Collect held-out metrics from every ``summary.json`` under a runs directory.

    python scripts/collect_results.py runs            # prints Markdown tables
    python scripts/collect_results.py runs --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = {
    "classification": ["accuracy", "mean_auroc", "macro_f1"],
    "regression": ["mae_dex", "rmse_dex", "r2", "pearson_r"],
    "super_resolution": ["psnr", "ssim", "mae", "mse"],
}


def collect(runs: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(runs.glob("*/*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if "test" not in summary:
            continue
        config = json.loads((summary_path.parent / "config.json").read_text())
        test = summary["test"]
        rows.append(
            {
                "task": config["task"],
                "run": summary_path.parent.name,
                "backbone": config["backbone"],
                "adaptation": config["adaptation"] if config["backbone"].startswith("vit") else "-",
                "train": config["train_dataset"],
                "shots": "full" if config["shots"] is None else config["shots"],
                "test": Path(test["test_root"]).name,
                **{key: test.get(key) for key in COLUMNS[config["task"]]},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = collect(args.runs)
    for task, metrics in COLUMNS.items():
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        header = ["run", "backbone", "adaptation", "train", "shots", "test", *metrics]
        print(f"\n### {task}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
        for row in task_rows:
            cells = [f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in header]
            print("| " + " | ".join(cells) + " |")
    if args.csv and rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
