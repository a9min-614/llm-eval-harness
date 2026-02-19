"""Save results to JSON / markdown and print a summary table."""

import json
import os
from collections import defaultdict
from datetime import datetime

from tabulate import tabulate


def _compute_accuracy(results: list[dict]) -> dict[tuple, dict]:
    """Aggregate accuracy by (model, subject)."""
    buckets: dict[tuple, list] = defaultdict(list)
    for r in results:
        key = (r["model"], r["subject"])
        buckets[key].append(r["correct"])

    return {
        key: {
            "n": len(vals),
            "correct": sum(vals),
            "accuracy": sum(vals) / len(vals) if vals else 0.0,
        }
        for key, vals in buckets.items()
    }


def _build_table_rows(stats: dict[tuple, dict]) -> list[list]:
    rows = []
    for (model, subject), s in sorted(stats.items()):
        rows.append([model, subject, f"{s['accuracy']:.1%}", s["correct"], s["n"]])
    return rows


TABLE_HEADERS = ["Model", "Subject", "Accuracy", "Correct", "N"]


def print_summary_table(all_results: list[dict]) -> None:
    """Print a formatted accuracy table to stdout."""
    stats = _compute_accuracy(all_results)
    rows = _build_table_rows(stats)
    print("\n" + tabulate(rows, headers=TABLE_HEADERS, tablefmt="github") + "\n")


def save_results(all_results: list[dict], output_dir: str) -> str:
    """Save raw results as JSON; return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{ts}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    return path


def save_markdown_report(all_results: list[dict], output_dir: str) -> str:
    """Save a markdown summary report; return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    stats = _compute_accuracy(all_results)
    rows = _build_table_rows(stats)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"{ts}.md")

    models = sorted({r["model"] for r in all_results})
    subjects = sorted({r["subject"] for r in all_results})
    total_samples = len(all_results)

    with open(path, "w") as f:
        f.write("# LLM Eval Harness — Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Models:** {', '.join(models)}  \n")
        f.write(f"**Subjects:** {', '.join(subjects)}  \n")
        f.write(f"**Total samples evaluated:** {total_samples}  \n\n")
        f.write("## Accuracy by Model and Subject\n\n")
        f.write(tabulate(rows, headers=TABLE_HEADERS, tablefmt="github"))
        f.write("\n")

    return path
