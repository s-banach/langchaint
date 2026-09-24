LiteLLM lists these OpenAI or Anthropic direct-API text models, which langchaint neither prices nor ignores.

For each model, either add it to `OPENAI_LITELLM_KEYS` or `ANTHROPIC_LITELLM_KEYS` in `scripts/update_pricing_metadata.py` and rerun `uv run python -m scripts.update_pricing_metadata`, or add it to `scripts/pricing/ignored-litellm-model-keys.json`.
The monthly refresh updates this issue, comments when the list changes, and closes the issue once no model remains.

