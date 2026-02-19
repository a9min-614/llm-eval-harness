# LLM Eval Harness

A benchmark harness that evaluates multiple OpenAI models on [MMLU](https://huggingface.co/datasets/cais/mmlu) (Massive Multitask Language Understanding).

Demonstrates key OpenAI SDK patterns:

- **`openai.AsyncOpenAI()`** — async client for concurrent requests
- **Structured outputs** (`response_format` with JSON schema) — guaranteed A/B/C/D extraction, no regex needed
- **`asyncio.Semaphore`** — respects concurrency limits to stay within rate limits
- **`tenacity`** — exponential backoff retry on `RateLimitError`

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Smoke test (5 samples, 1 subject, cheapest model only)
python run_eval.py --n-samples 5 --subjects high_school_mathematics --models gpt-4o-mini

# 4. Full run (~$0.10 in API costs)
python run_eval.py
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--models` | `gpt-4o-mini gpt-4o gpt-3.5-turbo` | Model IDs to evaluate |
| `--subjects` | `high_school_mathematics college_computer_science professional_law` | MMLU subjects |
| `--n-samples` | `50` | Max samples per subject |
| `--concurrency` | `10` | Max concurrent API requests |
| `--output-dir` | `results/` | Directory for output files |

## Example Output

```
| Model         | Subject                      | Accuracy | Correct | N  |
|---------------|------------------------------|----------|---------|----|
| gpt-3.5-turbo | college_computer_science     | 68.0%    | 34      | 50 |
| gpt-3.5-turbo | high_school_mathematics      | 54.0%    | 27      | 50 |
| gpt-3.5-turbo | professional_law             | 52.0%    | 26      | 50 |
| gpt-4o        | college_computer_science     | 84.0%    | 42      | 50 |
| gpt-4o        | high_school_mathematics      | 78.0%    | 39      | 50 |
| gpt-4o        | professional_law             | 74.0%    | 37      | 50 |
| gpt-4o-mini   | college_computer_science     | 76.0%    | 38      | 50 |
| gpt-4o-mini   | high_school_mathematics      | 70.0%    | 35      | 50 |
| gpt-4o-mini   | professional_law             | 64.0%    | 32      | 50 |
```

Results are also saved to `results/` as `.json` and `.md` files.

## Project Structure

```
llm-eval-harness/
├── run_eval.py          # CLI entry point (argparse)
├── harness/
│   ├── benchmark.py     # MMLU loader
│   ├── evaluator.py     # Async OpenAI calls + structured output
│   └── reporter.py      # JSON / markdown output + console table
├── results/             # Output directory (gitignored except .gitkeep)
├── requirements.txt
└── .env.example
```

## How It Works

1. **`benchmark.py`** — Downloads MMLU via HuggingFace `datasets`, converts integer answer indices to letters (A/B/C/D), and randomly samples up to `n_samples` per subject.

2. **`evaluator.py`** — For each sample, formats a prompt and calls `client.chat.completions.create()` with `response_format={"type": "json_schema", ...}`. The JSON schema enforces `{"answer": "A"|"B"|"C"|"D"}`, eliminating parse errors. All calls are dispatched concurrently behind a `Semaphore`, and `tenacity` retries on `RateLimitError` with exponential backoff.

3. **`reporter.py`** — Aggregates accuracy by `(model, subject)`, renders a GitHub-flavored markdown table via `tabulate`, and writes timestamped `.json` and `.md` files to `results/`.
