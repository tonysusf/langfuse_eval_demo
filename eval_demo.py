"""Compare two airline support routing prompts in Langfuse."""

import json
import os
import sys
import time
from pathlib import Path

MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MIN_REQUEST_INTERVAL = 4
ROUTES = {"baggage", "refund", "booking_change", "clarify"}
SCORE_NAME = "route_accuracy"

BASELINE_PROMPT = """Route the airline support message to one of these labels:
baggage, refund, booking_change, clarify.
Respond with exactly one label and no other text."""

CANDIDATE_PROMPT = """Route the airline support message based on what the traveler wants:
- baggage: Help with bag allowance, fees, or missing or damaged luggage.
- refund: Getting money back, reversing a charge, or checking refund status.
- booking_change: Changing, rebooking, or canceling a reservation.
- clarify: The traveler gives no clear request or presents equally plausible options.

Follow the requested action. After a flight cancellation, a request for money
back goes to refund, a request for a new flight goes to booking_change, and an
unspecified request for help goes to clarify.
Respond with exactly one label and no other text."""


def load_cases(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases.json must contain a nonempty list of cases")

    case_ids = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be an object")
        message = case.get("input")
        route = case.get("expected_output")
        metadata = case.get("metadata")
        case_id = metadata.get("case_id") if isinstance(metadata, dict) else None
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"Case {index} must have a nonempty input")
        if not isinstance(route, str) or route not in ROUTES:
            raise ValueError(f"Case {index} has an unsupported expected route")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError(f"Case {index} must have a unique, nonempty case ID")
        case_ids.add(case_id)
    return cases


def score_route(output, expected_output):
    return float(isinstance(output, str) and output.strip().casefold() == expected_output)


def main():
    required_keys = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENROUTER_API_KEY")
    missing = [key for key in required_keys if not os.environ.get(key, "").strip()]
    if missing:
        print(f"Set the following environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        cases = load_cases(Path(__file__).with_name("cases.json"))
        from langfuse import Evaluation, get_client
        from langfuse.openai import OpenAI
        from openai import APIError
    except (OSError, ValueError, ImportError) as error:
        print(f"Cannot start the demo: {error}", file=sys.stderr)
        return 1

    langfuse = get_client()
    next_request_at = time.monotonic()

    def evaluate(*, output, expected_output, **kwargs):
        return Evaluation(name=SCORE_NAME, value=score_route(output, expected_output))

    def run_prompt(name, prompt, client):
        def route_message(*, item, **kwargs):
            nonlocal next_request_at
            time.sleep(max(0, next_request_at - time.monotonic()))
            next_request_at = time.monotonic() + MIN_REQUEST_INTERVAL
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": item["input"]},
                    ],
                    temperature=0,
                )
            except APIError as error:
                case_id = item["metadata"]["case_id"]
                status = getattr(error, "status_code", None)
                failure = f"HTTP {status}" if status else type(error).__name__
                print(f"{name} / {case_id}: OpenRouter request failed ({failure})", file=sys.stderr)
                raise
            return response.choices[0].message.content if response.choices else None

        print(f"Running {name} with {MODEL}...", flush=True)
        result = langfuse.run_experiment(
            name=name,
            description="Airline support routing prompt comparison",
            metadata={"model": MODEL, "prompt_version": name},
            data=cases,
            task=route_message,
            evaluators=[evaluate],
            max_concurrency=1,
        )
        print(result.format())
        experiment_url = getattr(result, "experiment_url", None) or result.dataset_run_url
        if experiment_url:
            print(experiment_url)

        scores = {
            item.item["metadata"]["case_id"]: evaluation.value
            for item in result.item_results
            for evaluation in item.evaluations
            if evaluation.name == SCORE_NAME
        }
        unscored = [case["metadata"]["case_id"] for case in cases if case["metadata"]["case_id"] not in scores]
        if unscored:
            print(f"{name} is incomplete. Unscored cases: {', '.join(unscored)}", file=sys.stderr)
            return None
        return sum(scores.values()) / len(cases)

    try:
        if not langfuse.auth_check():
            print("Langfuse authentication failed. Check the project keys and URL.", file=sys.stderr)
            return 1
        with OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=OPENROUTER_URL,
            max_retries=0,
            timeout=30,
        ) as client:
            baseline = run_prompt("baseline-v1", BASELINE_PROMPT, client)
            candidate = run_prompt("candidate-v1", CANDIDATE_PROMPT, client)
        if baseline is None or candidate is None:
            return 1
        print(f"Route accuracy | baseline {baseline:.1%} | candidate {candidate:.1%}")
        return 0
    except Exception as error:
        print(f"Comparison incomplete ({type(error).__name__}). Check the API configuration and logs.", file=sys.stderr)
        return 1
    finally:
        langfuse.flush()


if __name__ == "__main__":
    sys.exit(main())
