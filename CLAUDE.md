# langchaint

langchaint is an opinionated, provider-neutral Python client for LLM applications.
The API is alpha and may change without notice.

## Documentation

Keep CLAUDE.md to cross-module principles and architecture required to edit langchaint safely.
Put symbol behavior in the implementing code's docstring.
Never cite internal documents, design deliberation, dead alternatives, or prior code.
State current behavior and its reason without requiring historical context.
Document each exception relevant to the public interface and its condition in `Raises:`, regardless of where it originates.
Omit incidental exceptions and source details that do not affect caller handling.
Verify an SDK fact from the installed SDK before writing dependent code.
Put a verified SDK fact only in a docstring where the caller acts on the outcome.
Include the SDK version when an SDK fact can drift.
Document every public parameter and cross-provider difference.

## Vocabulary

- Call the project "langchaint".
- Use "package" only for its Python meanings.
- Call an `Adapter` implementation an "adapter".
- Call anthropic, openai, and model-serving platforms "providers".
- Compose a concrete adapter name from its provider and `Adapter`, as in `AnthropicMessagesAdapter`.
- Name each backend class for its provider.
- Compose a Bedrock class name with the model provider.
- Allow one adapter to report different `provider_name` values for direct and Bedrock clients.
- Use neutral vocabulary when providers disagree, such as `ToolCall` instead of `ToolUse`.
- Give a keyword and the variable passed to it one name, as in `tool_manager=tool_manager`.
- Put units and encodings in names, such as `cost_in_usd`, `elapsed_seconds`, and the `_json` suffix.
- Prefix related fields so sorting and completion group them, as in `input_tokens_*`, `generate_one`, and `generate_many`.
- Do not repeat the holder in an attribute name: write `tool.name`, not `tool.tool_name`.
- Use the full name for a cross-object reference, such as `tool_call_id` on `ToolMessage`.
- Give an interface the plain noun.
- Use `cache_breakpoint` for a user-placed prompt-cache boundary.
- `cache_breakpoint=True` means the reusable prompt prefix ends at that part.
- Never write bare `input_tokens` because providers count it differently.
- Use `input_tokens_cache_read`, `input_tokens_cache_write`, `input_tokens_cache_none`, and the derived `input_tokens_total`.
- Keep `content`, `output`, and `raw` distinct: model-facing message body, generation result payload, and unchanged provider data.
- Use `reasoning` only for reasoning the model produced.

## Application API

- Keep request execution, stream consumption, embeddings, and tool dispatch asynchronous.
- Run synchronous provider work through `cancellation.py` so cancellation waits for the work to settle.
- Leave agent loops and tool loops to applications.
- Make tool functions return data without control-flow signals.

## Requests and providers

- Create one `SharedBackoff` per rate-limit quota.
- Gate every request start through its `admitted()` block.
- Count `max_attempts` as requests sent, including the first.
- Disable SDK retries so langchaint accounts for every attempt.
- Wrap official SDK clients.
- Let the SDK assemble streams.
- Do not define wire `TypedDict` types.
- Send user inputs and model ids verbatim.
- Do not predict provider responses, probe endpoints, or add guards based on guessed provider rules.
- Raise client-side only for documented provider facts and detectable defects that would otherwise produce a silently wrong result.
- Keep SDKs as optional dependencies.
- Give each optional backend dependency the same lower bound in `[project.optional-dependencies]` and `[dependency-groups].dev`.
- Keep SDK imports out of the neutral core.
- Import each SDK at the backend subpackage module top under a guard that raises `ModuleNotFoundError` with installation instructions.
- Put the pricing source URL in each backend subpackage docstring.

## Results and errors

- Validate a structured response against the caller's model while preserving the response and billing.
- Classify provider failures into the retry loop's neutral actions.
- Retry transient failures during non-streaming generation and while opening a stream.
- Terminate the call on other provider failures.
- Return one outcome per `GenerationInput` without letting one non-transient failure cancel a sibling.
- Raise detectable binding defects before sending a request.
- Never return a parse without output as data.
- Preserve provider error text verbatim after a prefix that names the failure.
- Keep generated content out of `error_text` and `__str__` because tracing records both without regard to `capture_message_content`.
- Put recoverable content in its own field.
- Create a separate variant only when an outcome has different fields or changes control flow.
- Require variant-specific data as non-optional fields.
- Give each variant a defaulted `Literal` `kind` named from the class after dropping words shared by every variant.
- Match non-exception class variants on the string `.kind` attribute.
- The `.kind` attribute lets autocomplete provide the discriminator without imports of variant classes.
- Use `isinstance` for exceptions and builtin types.
- Re-emit every reasoning trace verbatim and in place across turns.
- Let applications trim reasoning.

## Usage and pricing

- Add a field to `Usage` only for a provider-invariant counter or a priced category that partitions request cost.
- Keep provider-specific details on the raw SDK usage.
- Require `usage` on every generation result and `GenerationError`.
- Derive totals from categories.
- Let applications carry their own fees.
- Make `usage` aggregate every available `Billing` across the call.
- Represent a nonzero category with no configured rate as NaN.
- Never fabricate prices or model catalogs.
- Use a provider subpackage's default rate table only when it maps the model id.
- Require caller-supplied `pricing` when the default rate table does not map the model id.
- Label provider-published list pricing as an estimate.
- Pass provider values through by reference.
- Construct a langchaint model only when its shape differs from the SDK object.

## Types and imports

- Use pydantic for a langchaint model only when serialization and validation justify it.
- State the validation benefit in each langchaint pydantic model docstring.
- Derive every langchaint pydantic model from `CheckedCopyModel`.
- Use a frozen dataclass or `NamedTuple` otherwise.
- Use runtime checks only for invalid values that a correctly typed argument can contain.
- Delete tests that suppress the type checker only to reach a runtime type check.
- Keep a `cast` only when an opaque value re-enters the typed API that serialized it or a langchaint value deliberately exceeds an SDK parameter type.
- Add a comment that names the boundary for every remaining `cast`.
- Applications import from top-level `langchaint` and backend subpackages.
- Adapter authors import from `langchaint.adapter` and `langchaint.conformance`.
- Top-level `__all__` re-exports only the SDK-free application surface.

## Tracing

- Keep OTel tracing in a guarded-import subpackage outside top-level `__all__`.
- Use OTel SDK configuration to enable, disable, and route tracing.
- Never make a span measure an event boundary that did not occur.
- Limit an attribute mapper's output to attribute names and values.
- Never pass the `GenerationInput` to an attribute mapper.
- Catch and log telemetry failures without propagating them.
- Require `capture_message_content` without a default for recording message content.
- Use OTel convention keys where available and `langchaint.*` otherwise.

## Module map

- `llm.py`: client binding and generation.
- `_config_fingerprint.py`: deterministic binding and generation-input fingerprints.
- `_generate_many_records.py`: validated JSON resume state and atomic result-record persistence.
- `adapter.py`: the SDK-free neutral adapter contract.
- `cancellation.py`: cancellation-safe synchronous provider work.
- `conformance.py`: SDK-free adapter invariants that adapter tests inherit.
- `embedding.py`: provider-neutral embedding execution and output validation.
- `shared_backoff.py`: request admission for one rate-limit quota.
- `exceptions.py`: the error types.
- `response.py`: generation results and tabular call and attempt views.
- `call.py`: attempt records, immutable call history, and retry accounting.
- `streaming.py`: the stream handle.
- `tools.py`: tool forms, dispatch, and dispatch outcomes.
- `messages.py`: provider-neutral messages, content parts, and JSON round trips.
- `usage.py`: token accounting and per-category costs.
- `checked_copy.py`: the base for langchaint pydantic models.
- `pricing.py`: SDK-free rate arithmetic and per-attempt `Billing`.
- `anthropic/`, `cohere/`, `deepseek/`, `gemini/`, `openai/`: backend subpackages that require their SDKs.
- `inference_params.py`: inference parameters.
- `run_many.py`: bounded execution of zero-argument async callables without langchaint imports.
- `sequence_not_str.py`: the sequence protocol that excludes bare `str` values.
- `tracing/`: the optional OTel subpackage.

## Checks

Trigger: before committing.
Run `scripts/CI.sh` until it reports zero errors.
Keep tests offline.
Use constructed SDK objects and stub adapters.
Never use API keys in tests.
