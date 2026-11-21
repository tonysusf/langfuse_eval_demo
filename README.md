# Airline support routing eval

Compare a baseline prompt with a candidate that defines the airline support routes and handles ambiguous requests. Both prompts classify the same 12 synthetic messages using [Google Gemma 4 26B A4B's free variant](https://openrouter.ai/google/gemma-4-26b-a4b-it:free) through OpenRouter. Langfuse records each prediction, its score, and the model call.

## Run the demo

Use Python 3.11 or newer. Create a Langfuse project and an [OpenRouter API key](https://openrouter.ai/keys). Set the Langfuse base URL to match your project region or self-hosted instance.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'langfuse>=4,<5' 'openai>=2,<3'

export LANGFUSE_PUBLIC_KEY='your-project-public-key'
export LANGFUSE_SECRET_KEY='your-project-secret-key'
export LANGFUSE_BASE_URL='https://cloud.langfuse.com'
export OPENROUTER_API_KEY='your-openrouter-api-key'

python eval_demo.py
```

The script spaces out its 24 model requests, so allow at least 90 seconds for a complete run. If you hit an HTTP 429, check your [OpenRouter free-tier limits](https://openrouter.ai/collections/free-models) before running it again.

## Compare the results

The terminal prints both run summaries and the overall route accuracy. Each case earns 1 when the full response matches its expected label after trimming whitespace and ignoring case. All other responses score 0.

In Langfuse, open **Experiments**, select **baseline-v1** and **candidate-v1**, and click **Compare**. Open a case where the scores differ and inspect its trace to see the prompt, traveler message, and prediction. Try editing the candidate prompt in `eval_demo.py` and rerunning it with the same `cases.json`. [Langfuse experiment comparison guide](https://langfuse.com/docs/evaluation/experiments/compare-experiments)
