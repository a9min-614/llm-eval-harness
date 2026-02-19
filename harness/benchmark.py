"""MMLU benchmark loader."""

import random
from datasets import load_dataset

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu_samples(subjects: list[str], n_samples: int, seed: int = 42) -> list[dict]:
    """Load MMLU samples for the given subjects.

    Args:
        subjects: List of MMLU subject names (e.g. "high_school_mathematics").
        n_samples: Max samples per subject to return.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys: subject, question, choices, answer_letter.
    """
    rng = random.Random(seed)
    all_samples = []

    for subject in subjects:
        print(f"  Loading MMLU subject: {subject}")
        dataset = load_dataset("cais/mmlu", subject, split="test")

        rows = list(dataset)
        if len(rows) > n_samples:
            rows = rng.sample(rows, n_samples)

        for row in rows:
            all_samples.append(
                {
                    "subject": subject,
                    "question": row["question"],
                    "choices": row["choices"],  # list of 4 strings
                    "answer_letter": LETTER_MAP[row["answer"]],  # A/B/C/D
                }
            )

    return all_samples
