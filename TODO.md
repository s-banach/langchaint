# TODO

Work found by comparing langchaint against the 24 example programs in the pydantic-ai repository.
Each item states what to do and how to know it is done.

## No binding sends a provider's built-in tool

Every adapter populates the wire `tools` key itself and refuses it in `extra_body`, so an entry for a provider's own tool cannot reach that array.
One such tool is reachable, and only on the Chat Completions adapter.
`web_search_options` is a Chat Completions request key that adapter does not populate, so `extra_body={"web_search_options": {...}}` binds and sends.
`example-provider-tools.md` sketches the binding parameter that would open the other three.

### The plan

1. Add `provider_tools` to `bind` and to `Binding`: opaque wire entries, defaulting to none, sent verbatim.
2. Append each entry to the `tools` array each adapter builds, after the entries `tool_schemas` produced.
3. Document that an entry carries the tool's configuration and no query.
   openai 2.51.0's `WebSearchToolParam` has `type`, `filters`, `search_context_size`, and `user_location`, and the model writes the query from the user message.
4. Document that nothing dispatches through `ToolManager`: the provider runs the tool, so no `ToolCall` reaches the application.

Stop when a bound entry appears in the request each adapter builds, checked offline against the built request.

## A provider-run tool charges a fee no Usage field holds

Two cases, verified against the installed SDKs.
anthropic reports `usage.server_tool_use.web_search_requests` and `web_fetch_requests`, which are request counts (anthropic 0.120.2).
openai reports neither, on `ResponseUsage` and `CompletionUsage` alike (openai 2.51.0).
A per-invocation fee is not one of the four priced token categories, so it needs a fifth.

### The plan

1. `Usage` gains `provider_tool_cost_in_usd` and no counter.
   Per-tool counts are provider-shaped, so they stay on the raw usage beside it, where the uncollapsed cache-write counters already live.
   `cost_in_usd` sums five terms and `sum_of` folds the fifth.
2. Its value follows the rule `pricing.py` already states: `0.0` when no provider tool ran, the priced sum when every invocation langchaint counted had a rate, and NaN otherwise.
   Never zero for an unknown, which under-reports while looking exact.
3. Rates go per tool name in each backend's rate table, expressed per invocation.
   anthropic prices from `server_tool_use`, and the Responses adapter counts its built-in tool call items.
   Chat Completions can see from `annotations` that a search happened but cannot count calls, so it reports NaN.
4. Rewrite the `Usage` docstring paragraph saying server-side tool use has no cost.

`cost_in_usd` sums the categories, so a turn whose provider tool went unpriced reports a NaN total.

Undecided, and billing-relevant, so the user decides: whether the anthropic rate table ships `web_search` and `web_fetch` at their published per-request list price, called an estimate as the token tables are, or whether every provider-tool rate is the caller's to supply and every turn that used one reports NaN until they do.

Stop when a constructed anthropic usage reporting `web_search_requests` prices at the table's per-invocation rate, and reports NaN where the table holds no rate for that tool.

## Embeddings

The pydantic-ai RAG example needs embeddings, and langchaint generates text only.
