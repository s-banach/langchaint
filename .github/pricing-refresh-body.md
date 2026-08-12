This pull request refreshes LiteLLM pricing metadata.

Review the provider documentation before merging.

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
- [ ] Update `scripts/pricing/provider-pricing-metadata.json` when documentation changes.
- [ ] Rerun `uv run python -m scripts.update_pricing_metadata` after metadata changes.
- [ ] Rerun `scripts/CI.sh` after metadata changes.
