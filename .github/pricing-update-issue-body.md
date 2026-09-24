The monthly pricing refresh found LiteLLM pricing data that langchaint's committed pricing metadata does not reflect, so you need to update the pricing metadata.

1. For each model under **New models**, either price it or add it to `scripts/pricing/ignored-litellm-model-keys.json`. To price a model, edit `scripts/update_pricing_metadata.py`:
   - Add an OpenAI model to `OPENAI_LITELLM_KEYS`.
   - Add an Anthropic model to `ANTHROPIC_LITELLM_KEYS`, to `ANTHROPIC_INFERENCE_GEO_MODELS` when the Anthropic data-residency page applies US-only pricing to it, and its Bedrock key to `ANTHROPIC_BEDROCK_LITELLM_KEYS` when LiteLLM lists one.
   - Add an alias to `OPENAI_ALIASES` or `ANTHROPIC_ALIASES` when LiteLLM lists the alias key.
2. Run `uv run python -m scripts.update_pricing_metadata`.
3. Check the regenerated rates against the provider documentation:
   - [ ] Check [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
   - [ ] Check [OpenAI Fast mode](https://developers.openai.com/api/docs/guides/fast-mode).
   - [ ] Check [OpenAI Flex processing](https://developers.openai.com/api/docs/guides/flex-processing).
   - [ ] Check [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
   - [ ] Check [Anthropic service tiers](https://platform.claude.com/docs/en/api/service-tiers).
   - [ ] Check [Anthropic data residency](https://platform.claude.com/docs/en/manage-claude/data-residency).
   - [ ] Check tool invocation rates.
   - [ ] Check public `service_tier` rates and mappings.
   - [ ] Check long-context thresholds and multipliers.
   - [ ] Check regional pricing multipliers.
   - [ ] Check model aliases.
   - [ ] Check Anthropic Bedrock model mappings.
   - [ ] Check for new billing categories.
4. When the documentation differs, update `scripts/pricing/provider-pricing-metadata.json` and rerun `uv run python -m scripts.update_pricing_metadata`.
5. Run `scripts/CI.sh` and commit the result.

The monthly refresh updates this issue, comments when the listed rate changes or new models change, and closes the issue once nothing needs an update.
