#!/usr/bin/env python3
"""CLI entry point for the LLM Eval Harness."""

import argparse
import asyncio
import os

import openai
from dotenv import load_dotenv

from harness.benchmark import load_mmlu_samples
from harness.evaluator import run_model_eval
from harness.reporter import print_summary_table, save_markdown_report, save_results

DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
DEFAULT_SUBJECTS = [
    "high_school_mathematics",
    "college_computer_science",
    "professional_law",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI models on the MMLU benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="OpenAI model IDs to evaluate.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        metavar="SUBJECT",
        help="MMLU subject names.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        metavar="N",
        help="Max samples per subject.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Max concurrent OpenAI API requests.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        metavar="DIR",
        help="Directory to write result files.",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    print(f"\nLLM Eval Harness")
    print(f"  Models:     {', '.join(args.models)}")
    print(f"  Subjects:   {', '.join(args.subjects)}")
    print(f"  Samples:    {args.n_samples} per subject")
    print(f"  Concurrency:{args.concurrency}")
    print()

    print("Loading MMLU samples...")
    samples = load_mmlu_samples(args.subjects, args.n_samples)
    print(f"  {len(samples)} samples loaded.\n")

    client = openai.AsyncOpenAI(api_key=api_key)
    all_results: list[dict] = []

    for model in args.models:
        print(f"Evaluating {model} ({len(samples)} samples)...")
        results = await run_model_eval(client, model, samples, args.concurrency)
        correct = sum(r["correct"] for r in results)
        print(f"  Done — {correct}/{len(results)} correct ({correct/len(results):.1%})")
        all_results.extend(results)

    print_summary_table(all_results)

    json_path = save_results(all_results, args.output_dir)
    md_path = save_markdown_report(all_results, args.output_dir)
    print(f"Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
