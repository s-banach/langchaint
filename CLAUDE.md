# langchaint

A provider-neutral LLM client library. Alpha: the API is unstable and may change without notice.

## Docstrings and comments

Never cite internal dev documents, decision logs, spec files, dead alternatives, or the repo's prior state.
Delete prose that depends on a reader knowing an earlier version.
A sound sentence is independent of history.

Module docstrings specify mechanics.
CLAUDE.md states cross-module principles.
Write only a cross-module rule, its criterion, and at most one edge-case example in CLAUDE.md.
Put behavior in the implementing code's docstring.
Update that docstring when behavior changes.
A CLAUDE.md rule must survive any refactor that preserves the design tenets.
Put a rule that names a symbol in that symbol's docstring unless the rule defines the name itself, such as `input_tokens`.

Keep only universally beneficial design rules in CLAUDE.md.
Put langchaint-specific choices in the relevant code docstring.

Never record design deliberation in the repo.
State current behavior and its reason.
For an absent feature with a user-facing answer, state what the reader should do.
Write nothing when no user-facing answer exists.

Document every exception a function can raise or propagate and its condition.
Use `Raises:` for direct raises and prose for propagated raises.
Enforce this in review because Ruff cannot detect missing or stale raises.

## Provider facts

Never assert provider behavior from memory.
Verify it by introspecting the installed `anthropic` and `openai` packages before writing or reviewing dependent code.

Put a verified provider fact in a docstring only where the caller acts on the outcome.
Name the outcome and omit the SDK mechanism.
Include the SDK version when the fact can drift.

## Naming rules

- Give the keyword and the variable passed to it one name: `tool_manager=tool_manager`.
- Call the project "langchaint", never "the package" or "the library".
- Use "package" only for its Python meanings.
- Call an `Adapter` implementation an "adapter".
- Call anthropic, openai, and model-serving platforms "providers".
- Compose each concrete adapter name from its provider and adapter, as in `AnthropicMessagesAdapter`.
- One adapter may report different `provider_name` values for direct and Bedrock clients.
- Use neutral vocabulary when providers disagree: choose the majority wire name or a neutral name (`ToolCall`, not `ToolUse`).
- Never write bare `input_tokens`: anthropic's field of that name excludes cache reads while openai's equivalent includes them. Use the partition `input_tokens_cache_read`/`_cache_write`/`_cache_none` and the derived `input_tokens_total`.
- Put units and encodings in names (`cost_in_usd`, `elapsed_seconds`); mark unparsed JSON text with the `_json` suffix.
- Prefix related fields so sorts and completions group them: `input_tokens_*`, `generate_one`, and `generate_many`.
- Do not repeat the holder in an attribute name (`tool.name`, not `tool.tool_name`).
- Use the full name for cross-object references (`tool_call_id` on `ToolMessage`).
- Give the interface the plain noun.
- Name each concrete form for its argument-spec technology or fixed behavior.
- Use `cache_breakpoint` as the neutral name for a user-placed prompt-cache boundary: True on a part means the reusable prompt prefix ends there.
- Count `max_attempts` as requests sent, including the first. Configure SDK clients with no internal retries.
- Keep `content`, `output`, and `raw` distinct: model-facing message body, generation result payload, and provider data langchaint passes through unchanged.
- Use `reasoning` only for reasoning the model produced.

## Design rules

- Leave the tool loop to the application.
- Ship no agent loop.
- Make tool functions return data, never control-flow signals.
- Create one `SharedBackoff` per rate-limit quota. Its `admitted()` block gates every request-start path.
- Wrap official SDK clients.
- Let the SDK assemble streams.
- Do not define wire TypedDicts.
- Validate a structured response against the caller's model while the response and its billing remain available.
- Classify errors only by retry behavior.
- Retry transient errors and propagate others.
- Return one outcome per `GenerationInput` without letting one non-transient failure cancel a sibling.
- Raise detectable binding defects before requests.
- Never return a parse with no output as data.
- Preserve provider error text verbatim after a prefix naming the failure.
- The provider text describes an unmodeled condition.
- Keep generated content out of `error_text` and `__str__` because tracing records both regardless of `capture_message_content`.
- Put recoverable content in its own field.
- Add a field to `Usage` only for a provider-invariant counter or priced category that partitions request cost.
- Keep provider-specific details on the raw SDK usage.
- Require `usage` on every carrier.
- Derive totals from categories.
- Let applications carry their own fees.
- Scope `usage` as the paid total across every attempt, on success and on failure.
- Set an unpriceable category to NaN.
- Never raise or use zero for an unknown cost.
- An exception would discard paid output, and zero would hide unknown cost.
- Never fabricate prices or model catalogs.
- Use default pricing only from a carried rate table.
- Require `pricing` when no rate table maps the model id.
- Label a first-party list price from a rate-setting platform an estimate.
- Pass provider values through by reference.
- Construct a langchaint model only when its shape differs from the SDK object.
- Represent branchable outcomes as a union of frozen dataclasses with one variant per outcome.
- Split variants only where their fields differ.
- Require all extra data as non-optional fields.
- Give each class variant a `Literal` `kind` defaulting to its snake-cased name after dropping words shared by every variant.
- Use `kind` for exhaustive matching without imports.
- Select builtin variants with `isinstance`.
- This shape narrows types without `cast` or nullable assertions.
- Do not split an element type consumed by folding.
- Re-emit every reasoning trace verbatim and in place across turns.
- Let applications trim reasoning.
- Send user inputs verbatim, including invalid inputs.
- Do not predict provider responses, probe endpoints, or add guards based on guessed provider rules.
- Do not restate this rule for individual cases.
- Raise client-side only for documented provider facts and detectable defects that would produce a silently wrong result.
- Use pydantic only when serde and validation justify it.
- Use a frozen dataclass or NamedTuple otherwise.
- State the validation benefit in each pydantic model docstring.
- Derive every pydantic model from the checked-copy base, which rejects keys that are not fields.
- Keep SDKs as optional dependencies.
- Declare each backend dependency set as an extra with the same lower bounds used by the dev group.
- Let applications add tighter SDK pins.
- Keep SDK imports out of the neutral core.
- Import each SDK at the backend subpackage module top under a guard that raises `ModuleNotFoundError` with installation instructions.
- Name each backend class for its provider.
- Compose Bedrock class names with the model provider.
- Send each model id verbatim.
- Document every parameter and cross-provider difference.
- Put the pricing source URL in each subpackage docstring.
- Applications import from top-level `langchaint` and backend subpackages.
- Adapter authors import from `langchaint.adapter` and `langchaint.conformance`.
- Top-level `__all__` re-exports only the SDK-free application surface.
- Use runtime checks only for errors a correctly typed argument can contain, such as an out-of-range value.
- A strict type checker handles argument types.
- A strict type checker accepts `True` as `int` because `bool` subclasses `int`.
- Delete tests that suppress the type checker only to reach a runtime type check.
- Ship thin OTel tracing in-tree as a guarded-import subpackage outside top-level `__all__`.
- Never make a span measure a fake event boundary.
- Give the mapper attribute names and values, never the `GenerationInput`.
- Catch and log telemetry failures.
- Never propagate telemetry failures.
- Always wrap and use OTel SDK configuration for enabling, disabling, and routing.
- Record message content only through `capture_message_content`, which is required and has no default.
- Use convention keys where available and `langchaint.*` otherwise.

## Module map

Use one line per module to state its purpose.
Use the module docstring to specify its contents.
Do not list symbols because inventories become stale.

- `llm.py`: the client `LLM` and the `BoundLLM` its `bind` returns.
- `adapter.py`: the neutral base contract, dual-audience; imports no SDK.
- `cancellation.py`: cancellation-safe execution of synchronous provider work.
- `conformance.py`: the invariants every adapter holds to, as a test class an adapter author inherits; imports no SDK and no test runner.
- `embedding.py`: provider-neutral embedding execution and output validation.
- `shared_backoff.py`: paced request admission for one rate-limit quota.
- `exceptions.py`: the error vocabulary.
- `response.py`: the generate results and their flattening to a calls table and an attempts table.
- `call.py`: the per-call history: one attempt's record, the frozen `CallRecord` every result carries, and the ledger the retry loops drive.
- `streaming.py`: the stream handle.
- `tools.py`: the tool forms, the `Tool` protocol, `ToolManager`, and the dispatch outcome types.
- `messages.py`: the provider-neutral message tree and content parts, and their JSON round trip.
- `usage.py`: token accounting and the per-category costs that travel with it.
- `checked_copy.py`: the base of langchaint's pydantic models.
- `pricing.py`: rate arithmetic and the `Billing` an attempt carries. It imports no SDK or error class. Each provider-shaped rate table lives in the backend subpackage whose adapter spends it.
- `anthropic/`, `cohere/`, `deepseek/`, `gemini/`, `openai/`: the backend subpackages; importing one requires its SDK.
- `inference_params.py`: the inference parameters.
- `run_many.py`: runs zero-argument async callables under a pending bound. It imports nothing from langchaint and models nothing about LLMs.
- `sequence_not_str.py`: the sequence protocol excluding bare `str` values.
- `tracing/`: the OTel subpackage; importing it requires opentelemetry-api, and it is off the top-level `__all__`.

## Checks

Trigger: before committing.
Run `scripts/CI.sh` until it reports zero errors.

`scripts/CI.sh` runs `pyrefly check`, `ruff check`, `ruff format --check`, and `pytest` through `uv run`.
`pyproject.toml` explains the de-selected docstring rules.
Keep tests offline with constructed SDK objects, stub adapters, and no API keys.

## Releasing

Trigger: releasing a version.
Bump `version` in `pyproject.toml` and push to `main`.
Never create a `v*` tag by hand.
`.github/workflows/publish.yml` publishes to PyPI and creates a tag only when `version` exceeds every existing `v*` tag.
A manual tag for the release version blocks the release.

Confirm with the user before pushing a commit that changes `version`.
PyPI does not accept a re-uploaded version.

# Casts

Redesign every `cast` away except at two boundaries.
Keep a `cast` when a deliberately opaque value re-enters the typed API that serialized it and redesign would reshape the payload.
Keep a `cast` when langchaint vocabulary is deliberately wider than the SDK literal under "Honor user inputs faithfully".
Keep each surviving `cast` on one line with a comment naming its boundary.
