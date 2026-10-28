The monthly pricing refresh found LiteLLM pricing data that langchaint's committed pricing metadata does not reflect, so you need to update the pricing metadata.

1. For each model under **New models**, either price it or add it to `scripts/pricing/ignored-litellm-model-keys.json`. To price a model, edit `scripts/update_pricing_metadata.py`:
   - Add an OpenAI model to `OPENAI_LITELLM_KEYS`.
   - Add an Anthropic model to `ANTHROPIC_LITELLM_KEYS`, to `ANTHROPIC_INFERENCE_GEO_MODELS` when the Anthropic data-residency page applies US-only pricing to it, and its Bedrock key to `ANTHROPIC_BEDROCK_LITELLM_KEYS` when LiteLLM lists one. Add that Bedrock key to `AnthropicBedrockModelId` in `src/langchaint/anthropic/__init__.py`.
   - Add an alias to `OPENAI_ALIASES` or `ANTHROPIC_ALIASES` when LiteLLM lists the alias key.
   - Add an OpenAI embedding model to `OPENAI_EMBEDDING_LITELLM_KEYS`, and give it an overload and its dimension rule in `OpenAI.embedding_model` in `src/langchaint/openai/__init__.py`. When it rejects the `dimensions` request parameter or counts tokens with an encoding other than `cl100k_base`, also change `src/langchaint/openai/embedding_adapter.py`.
   - Add a Cohere embedding model on Bedrock by its unprefixed key, whether you price or ignore it; each cross-region key resolves through it. Add it to `COHERE_BEDROCK_EMBED_V4_LITELLM_KEYS` when it accepts the `CohereEmbedV4Dimension` values as `output_dimension` with 1536 as default, and to `COHERE_BEDROCK_EMBED_V3_LITELLM_KEYS` when it accepts no `output_dimension` and returns 1024 dimensions. Otherwise extend `CohereBedrock.embedding_model` in `src/langchaint/cohere/__init__.py`.
2. Run `uv run python -m scripts.update_pricing_metadata`.
3. Check the regenerated rates against the provider documentation:
   - [ ] Check [OpenAI pricing](https://developers.openai.com/api/docs/pricing).
   - [ ] Check [OpenAI Fast mode](https://developers.openai.com/api/docs/guides/fast-mode).
   - [ ] Check [OpenAI Flex processing](https://developers.openai.com/api/docs/guides/flex-processing).
   - [ ] Check [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
   - [ ] Check [Anthropic service tiers](https://platform.claude.com/docs/en/api/service-tiers).
   - [ ] Check [Anthropic data residency](https://platform.claude.com/docs/en/manage-claude/data-residency).
   - [ ] Check [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for Cohere embedding rates.
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
