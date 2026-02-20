"""Async evaluator supporting OpenAI and Anthropic models with rate-limit retry."""

import asyncio
import json

import anthropic
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# JSON schema that forces OpenAI models to reply with only A, B, C, or D.
ANSWER_SCHEMA = {
    "name": "mmlu_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["A", "B", "C", "D"],
                "description": "The letter of the correct answer choice.",
            }
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

# OpenAI models that support json_schema structured outputs
_JSON_SCHEMA_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"}

# Claude model prefix for routing
_CLAUDE_PREFIX = "claude-"


def _openai_response_format(model: str) -> dict:
    if any(model.startswith(m) for m in _JSON_SCHEMA_MODELS):
        return {"type": "json_schema", "json_schema": ANSWER_SCHEMA}
    return {"type": "json_object"}


def _format_prompt(sample: dict) -> str:
    """Build the user message for a single MMLU question."""
    choices_text = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip("ABCD", sample["choices"])
    )
    return (
        f"Question: {sample['question']}\n\n"
        f"{choices_text}\n\n"
        "Reply with the letter (A, B, C, or D) of the correct answer."
    )


def _is_retryable_openai_rate_limit(exc: BaseException) -> bool:
    if not isinstance(exc, openai.RateLimitError):
        return False
    return getattr(exc, "code", None) != "insufficient_quota"


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=_is_retryable_openai_rate_limit,
    reraise=True,
)
async def _call_openai_api(client: openai.AsyncOpenAI, model: str, prompt: str) -> str:
    """Call the OpenAI chat API with structured output; returns predicted letter."""
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant taking a multiple-choice exam. "
                    'Respond with JSON in the format {"answer": "X"} where X is A, B, C, or D.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=_openai_response_format(model),
        temperature=0,
    )
    content = response.choices[0].message.content
    return json.loads(content)["answer"]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(anthropic.RateLimitError),
    reraise=True,
)
async def _call_claude_api(client: anthropic.AsyncAnthropic, model: str, prompt: str) -> str:
    """Call the Anthropic Claude API; returns predicted letter."""
    response = await client.messages.create(
        model=model,
        max_tokens=16,
        system=(
            "You are a knowledgeable assistant taking a multiple-choice exam. "
            'Respond with JSON in the format {"answer": "X"} where X is A, B, C, or D.'
        ),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.content[0].text
    return json.loads(content)["answer"]


async def evaluate_sample(
    openai_client: openai.AsyncOpenAI,
    anthropic_client: anthropic.AsyncAnthropic | None,
    model: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Evaluate a single sample; returns sample dict augmented with prediction."""
    async with semaphore:
        prompt = _format_prompt(sample)
        try:
            if model.startswith(_CLAUDE_PREFIX) and anthropic_client:
                predicted = await _call_claude_api(anthropic_client, model, prompt)
            else:
                predicted = await _call_openai_api(openai_client, model, prompt)
        except Exception as exc:
            predicted = "ERROR"
            print(f"    [WARN] {model} failed on sample: {exc}")

        return {
            **sample,
            "model": model,
            "predicted": predicted,
            "correct": predicted == sample["answer_letter"],
        }


async def run_model_eval(
    openai_client: openai.AsyncOpenAI,
    model: str,
    samples: list[dict],
    concurrency: int,
    anthropic_client: anthropic.AsyncAnthropic | None = None,
) -> list[dict]:
    """Evaluate all samples for a single model concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [evaluate_sample(openai_client, anthropic_client, model, s, semaphore) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)
