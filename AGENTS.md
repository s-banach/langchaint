# langchaint

langchaint is an opinionated, provider-neutral Python client for LLM applications.
The API is alpha and may change without notice.

## Documentation

Keep AGENTS.md to cross-module principles and architecture required to edit langchaint safely.
Put symbol behavior in the implementing code's docstring.
Never cite internal documents, design deliberation, dead alternatives, or prior code.
State current behavior and its reason without requiring historical context.
Document each exception relevant to the public interface and its condition in `Raises:`, regardless of where it originates.
Omit incidental exceptions and source details that do not affect caller handling.
Verify an SDK fact from the installed SDK before writing dependent code.
Put a verified SDK fact only in a docstring where the caller acts on the outcome.
Include the SDK version when an SDK fact can drift.
Document every public parameter and cross-provider difference.

## Terms

- langchaint: this project.
- provider: anthropic, openai, or a model-serving platform.
- adapter: an `Adapter` implementation.
- request: one send to the provider. Each retry is another request.
- response: the provider's reply to one request.
- input: one `GenerationInput`.
- request params: what every request for one input sends.
- assistant message: the `AssistantMessage` in one response.
- output: what application code reads from a finished assistant message, either its joined text or a `response_format` instance validated from its text. An assistant message gives output when that value exists.
- usable: an assistant message that gives output or has tool calls.
- kept assistant message: the usable assistant message that ends handling an input.
- generation: what handling an input produces when it succeeds. `GenerationError` is its failure.
- outcome: one way that handling an input, reading one response, or dispatching a tool call ends. An `*Outcome` type is the union of them.
- Use these terms, and never give one of these concepts a second name.

## Vocabulary

- Use "package" only for its Python meanings.
- Compose a concrete adapter name from its provider and `Adapter`, as in `AnthropicMessagesAdapter`.
- Name each backend class for its provider.
- Compose a Bedrock class name with the model provider.
- Allow one adapter to report different `provider_name` values for direct and Bedrock clients.
- Use neutral vocabulary when providers disagree, such as `ToolCall` instead of `ToolUse`.
- Give a keyword and the variable passed to it one name, as in `tool_manager=tool_manager`.
- Put units and encodings in public names and in values that cross modules, such as `cost_in_usd`, `elapsed_seconds`, and the `_json` suffix.
- A holder named for its unit, such as `usd_per_million_tokens`, covers its fields.
- A rate class whose docstring states the unit, such as `OpenAIRates`, covers its fields.
- Prefix related fields so sorting and completion group them, as in `input_tokens_*`, `generate_one`, and `generate_many`.
- Do not repeat the holder in an attribute name: write `tool.name`, not `tool.tool_name`.
- Use the full name for a cross-object reference, such as `tool_call_id` on `ToolMessage`.
- Give an interface the plain noun.
- Call every prompt-cache boundary a cache breakpoint, whether a part with `cache_breakpoint=True` places it or the adapter places it automatically, as with `automatic_cache_breakpoints`.
- `cache_breakpoint=True` means the reusable prompt prefix ends at that part.
- Never write bare `input_tokens` because providers count it differently.
- Use `input_tokens_cache_read`, `input_tokens_cache_write`, `input_tokens_cache_none`, and the derived `input_tokens_total`.
- Keep `content`, `output`, and `raw` distinct: model-facing message body, `Generation` payload, and unchanged provider data.
- Replay `assistant_message`, never `output`.
- Use `reasoning` only for reasoning the model produced.

## Application API

- Keep request execution, stream consumption, embeddings, and tool dispatch asynchronous.
- Run synchronous provider work through `concurrency/cancellation.py` so cancellation waits for the work to settle.
- Leave agent loops and tool loops to applications.
- Make tool functions return data without control-flow signals.

## Requests and providers

- Create one `SharedBackoff` per rate-limit quota.
- Gate every request start through its `admitted()` block.
- Disable SDK retries so langchaint accounts for every request.
- Wrap official SDK clients.
- Let the SDK assemble streams.
- Do not define wire `TypedDict` types.
- Send model ids, `system_prompt`, and every message exactly as given, including a replayed assistant message.
- When a provider rejects a message the user wrote, return the provider's error.
- Never build a `TextPart` from empty provider text, because Anthropic rejects an empty text block on replay.
- Applications replay an assistant message to continue after its tool calls, so dropping empty text leaves no replayed assistant message without parts.
- An assistant message without tool calls ends a tool loop, and an application that continues after one chooses what to send.
- Do not predict provider responses, probe endpoints, or add guards based on guessed provider rules.
- Raise client-side only for documented provider facts and detectable defects that would otherwise produce a silently wrong result.
- Keep SDKs as optional dependencies.
- Give each optional backend dependency the same lower bound in `[project.optional-dependencies]` and `[dependency-groups].dev`.
- Keep SDK imports out of the neutral core.
- Import each SDK at the backend subpackage module top under a guard that raises `ModuleNotFoundError` with installation instructions.
- Put the pricing source URL in each backend subpackage docstring.

## Outcomes and errors

- Validate a structured response against the caller's model while preserving the response and billing.
- Map each failed request to a neutral `RequestFailure`, which decides the retry and the rate-limit quota pause.
- Retry transient failures in `generate_one`, `generate_many`, and `generate_many_records`, and while `stream_one` opens a stream.
- Stop handling the input on other provider failures.
- Return one outcome per `GenerationInput` without letting one non-transient failure cancel a sibling.
- Raise detectable binding defects before sending a request.
- Never return a parse without output as data.
- Preserve provider error text verbatim after a prefix that names the failure.
- Keep generated content out of `error_text` and `__str__` because tracing records both without regard to `capture_message_content`.
- Put recoverable content in its own field.
- Create a separate variant only when an outcome has different fields or changes control flow.
- Require variant-specific data as non-optional fields.
- A variant may cover several outcomes with the same fields, and then validates the data each outcome requires.
- Give each variant a `Literal` `kind`.
- When a variant covers one outcome, default its `kind` and name it from the class after dropping words shared by every variant.
- Match non-exception class variants on the string `.kind` attribute.
- The `.kind` attribute lets autocomplete provide the discriminator without imports of variant classes.
- Use `isinstance` for exceptions and builtin types.
- Re-emit every reasoning trace verbatim and in place when replaying assistant messages.
- Let applications trim reasoning.

## Usage and pricing

- Add a field to `Usage` only for a provider-invariant counter or a priced category that partitions request cost.
- Keep provider-specific details on the raw SDK usage.
- Require `usage` on every `Generation` and `GenerationError`.
- Derive totals from categories.
- Let applications carry their own fees.
- Make `usage` aggregate every available `Billing` across the input's requests.
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
- Write a non-finite float constant as `float("nan")`, `float("inf")`, or `float("-inf")`.
- Import `math` as a module and reach its functions through it, as in `math.isnan`.
- Keep a `cast` only when an opaque value re-enters the typed API that serialized it or a langchaint value deliberately exceeds an SDK parameter type.
- Add a comment that names the boundary for every remaining `cast`.
- Applications import from top-level `langchaint` and backend subpackages.
- Adapter authors import from `langchaint.adapter` and `langchaint.conformance`.
- Top-level `__all__` re-exports only the SDK-free application surface.
- Keep source-module imports acyclic, including `TYPE_CHECKING` and function-local imports.
- Keep dependencies between sibling directories and root modules acyclic.
- Keep `common/` independent of other langchaint directories and root modules.
- Run `uv run python -m scripts.check_architecture` to check explicit source imports for these dependency rules.

## Tracing

- Keep OTel tracing in a guarded-import subpackage outside top-level `__all__`.
- Connect tracing to the core only through the SDK-free `Observer`, so application code is the same with or without tracing.
- A backend constructor passes its `observer` to every `LLM` it creates, and `LLM(observer=...)` and `ToolManager(observer=...)` accept one directly.
- Use OTel SDK configuration to enable, disable, and route tracing.
- Never make a span measure an event boundary that did not occur.
- Limit an attribute mapper's output to attribute names and values.
- Never pass the `GenerationInput` to an attribute mapper.
- Catch and log telemetry failures without propagating them.
- Require `capture_message_content` without a default for recording message content.
- Use OTel convention keys where available and `langchaint.*` otherwise.

## Module map

- `generation/llm.py`: client binding, the generate methods, and shared batch coordination.
- `generation/_config_fingerprint.py`: deterministic binding and generation-input fingerprints.
- `generation/_generate_many_records.py`: validated JSON resume state and atomic outcome-record persistence.
- `adapter.py`: the SDK-free neutral adapter contract and `ResponseIdentity`.
- `concurrency/cancellation.py`: cancellation-safe synchronous provider work.
- `conformance.py`: SDK-free adapter invariants that adapter tests inherit.
- `embedding.py`: provider-neutral embedding execution and output validation.
- `concurrency/shared_backoff.py`: request admission for one rate-limit quota.
- `common/exceptions.py`: basic shared exceptions without langchaint imports.
- `common/request_failure.py`: what a failed request means, and the mapping that generation and embedding share.
- `generation/errors.py`: normalized generation error records and live generation failures.
- `generation/response.py`: live generations, their records, and the outcome unions.
- `generation/tables.py`: tabular outcome and request views.
- `generation/request_history.py`: request records, immutable request history, and retry accounting.
- `generation/streaming.py`: the stream handle.
- `generation/observer.py`: the protocol that follows every input and every tool dispatch.
- `common/observed_operation.py`: the handle an observer returns for one input or one tool dispatch, and the guard that logs observer failures, without langchaint imports.
- `tools.py`: tool forms, dispatch, dispatch outcomes, and tool exceptions.
- `common/messages.py`: provider-neutral messages, content parts, and JSON round trips.
- `billing/usage.py`: token accounting and per-category costs.
- `common/checked_copy.py`: the base for langchaint pydantic models.
- `billing/pricing.py`: SDK-free rate arithmetic and per-request `Billing`.
- `anthropic/`, `cohere/`, `deepseek/`, `gemini/`, `openai/`: backend subpackages that require their SDKs.
- `concurrency/run_many.py`: bounded execution of zero-argument async callables without langchaint imports.
- `common/sequence_not_str.py`: the sequence protocol that excludes bare `str` values.
- `tracing/`: the optional OTel subpackage and its `OtelObserver`.
- `span_parsing.py`: OTel chat and execute_tool span parsing, and chat span conversion, without OpenTelemetry dependencies.

## Checks

Trigger: before committing.
Run `scripts/CI.sh` until it reports zero errors.
Keep tests offline.
Use constructed SDK objects and stub adapters.
Never use API keys in tests.
