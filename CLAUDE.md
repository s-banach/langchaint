# langchaint

A provider-neutral LLM client library. Alpha: the API is unstable and may change without notice.

## Docstrings and comments

Never cite internal dev documents, decision logs, spec files, or dead alternatives to the live code.
Never refer to the repo's own prior state. Diff-relative prose is a sentence that only makes sense to a reader who saw the change that introduced it: wording like "as before", "no longer", or "now", a reference to a state that exists nowhere in the current tree, or a justification for a question no reader of the file would ask. The test: a sound sentence reads the same whether the code was born in its current shape or arrived there by refactor.

Module docstrings are the spec of record for mechanics; CLAUDE.md is the spec of record for principles. Write in CLAUDE.md only a cross-module rule, its criterion, and at most one edge-case example per rule; keep how a behavior works in the docstring of the code that implements it, and when such a behavior changes, update that docstring, not CLAUDE.md. The durability test: a CLAUDE.md sentence reads the same after any refactor that preserves the design tenets, so a sentence naming a symbol belongs in that symbol's docstring, except where the rule is about that name (`input_tokens`).
A design rule earns its place here only if it is universally beneficial, so that a reader who had never seen langchaint would still call it correct. A choice that is merely the one langchaint made, including a feature deliberately not built, is not a design rule: it belongs in the docstring of the code a reader would reach for it in.
Never record a design deliberation anywhere in the repo: no sentence of the form "X is rejected", "X was considered", or "we chose this over X". State what the code does and why. Where a feature langchaint does not have has a user-facing answer, write that answer as what the reader should do instead, and write nothing when there is none.

Document what a function raises. In every function whose body can raise (directly, by re-raising, or by propagating a documented raise from a helper), name the exception types a caller may see and the condition for each, in a `Raises:` section when the raise is direct or in prose when it is not. Enforce this in review, not lint: the de-selected ruff docstring rules cannot catch a missing or stale raise (see the note in `pyproject.toml`).

## Provider facts

Never assert provider behavior (wire parameters, usage-field semantics, exception taxonomies, cache rules) from memory; verify it against the installed `anthropic`/`openai` packages by introspection before writing or reviewing code that depends on it.

Put a verified fact in a docstring only where the caller acts on it, naming the outcome the caller handles and not the SDK mechanism reaching it, with the SDK version when it could drift.

## Naming rules

- Give the keyword and the variable passed to it one name: `tool_manager=tool_manager`.
- Call the project "langchaint", never "the package" or "the library"; use "package" only for its Python meanings.
- Say "adapter" for an implementation of the class `Adapter`, including in compounds, and "provider" for anthropic and openai themselves and for a platform serving their models; a concrete name composes the two (`AnthropicMessagesAdapter`). "Provider" is wider than the company because the serving platform counts: one adapter reports a different `provider_name` over a direct client than over a Bedrock one.
- Prefer neutral over provider vocabulary: when providers disagree, take the majority wire name or a neutral one (`ToolCall` not `ToolUse`).
- Never write bare `input_tokens`: anthropic's field of that name excludes cache reads while openai's equivalent includes them. Use the partition `input_tokens_cache_read`/`_cache_write`/`_cache_none` and the derived `input_tokens_total`.
- Put units and encodings in names (`cost_in_usd`, `elapsed_seconds`); mark unparsed JSON text with the `_json` suffix.
- Use family prefixes to keep related fields adjacent in sorts and completions: `input_tokens_*`, `generate_one`/`generate_many` (arity in the suffix).
- Never stutter with the holder (`tool.name`, not `tool.tool_name`); carry the full name on cross-object references (`tool_call_id` on `ToolMessage`).
- Give the plain noun to the interface, because the protocol name is read far more often than any concrete name is written; name concrete forms by the technology their argument spec is written in, except a form distinguished by its fixed behavior, which is named for that behavior.
- Use `cache_breakpoint` as the neutral name for a user-placed prompt-cache boundary: True on a part means the reusable prompt prefix ends there.
- Count `max_attempts` as requests sent including the first, so 1 means no retrying; configure the SDK client so it never retries beneath langchaint and the count stays true.
- Keep `content`, `output`, and `raw` three concepts, never one word: a model-facing message body, the generation result payload, and provider data langchaint models nothing inside and hands back unchanged. `reasoning` names what the model produced, never a fourth name for one of those three.

## Design rules

- Never choose a billing-relevant configuration for the user: `automatic_prompt_caching` is a required keyword with no default (an unstated `False` is a billing choice as real as opting in), and any convenience on top of user-stated caching is opt-in and default-off. Honor user-placed `cache_breakpoint` marks under either binding value.
- Leave the tool loop to the application: ship no agent loop, and make a tool function return data, never a control-flow signal.
- Keep one `RateLimiter` owning retrying and pacing: one instance is one shared budget for the account it guards, gating every request-start path.
- Wrap official SDK clients and delegate stream assembly to the SDK. Write no wire TypedDicts by hand. Validate a structured response against the caller's model where the response is in scope, because an SDK that validates inside the call returning the response raises where neither the response nor its billing is reachable.
- Give the error taxonomy one axis, retry: retry a transient error and propagate the rest. Every non-transient error is one item's failure row, so a batch returns one outcome per `GenerationInput` and no item's failure cancels a sibling. A defect langchaint can detect in a binding raises before any request is sent. Never report a parse that returned no output as data.
- Put the provider's own error text in the error langchaint raises, unabridged: a prefix naming what failed, then the provider string. Never summarize, truncate, or replace it, because the provider's wording is the only description of a condition langchaint does not model. Keep generated content out of `error_text` and `__str__`: the tracing layer writes both into spans unconditionally, so content placed there escapes whatever `capture_message_content` the caller chose. Content a caller recovers from a failure goes on its own field.
- Admit a field to `Usage` only if it is a provider-invariant counter or one of the priced categories partitioning what a request cost; keep provider-specific detail on the raw SDK usage beside it, and never let `usage` be `None` on a carrier. A cost that is not one category's spend has no field: a total is derived from the categories, and an application's own fee is the application's to carry.
- Scope `usage` as the paid total across every attempt, on success and on failure.
- Price a category the rate table cannot price as NaN, never as an exception and never as zero: the response was paid for, so reporting an unknown cost must not destroy the output.
- Never fabricate a price or a model catalog: default pricing only from a carried rate table, require it where no table maps to the model id, and call a first-party list price on a rate-setting platform an estimate.
- Never take data out of an SDK object only to reconstruct it in a langchaint object of the same shape; pass provider values through by reference, constructing a langchaint model only where the shape genuinely changes.
- Discriminate outcomes by type, not a nullable flag: return a union of frozen dataclasses, one arm per outcome, extra data as required non-optional fields, so matching narrows and no consumer writes `cast` or `assert x is not None`. Every arm carries a `Literal` `kind` field defaulting to its own snake-cased name, dropping any word every arm shares, so a caller selects an arm without importing it and a match on `kind` is checked for exhaustiveness. An arm that is a builtin cannot hold a tag, so `isinstance` selects it; where more than one class arm remains, each still carries its own tag. Split only where fields genuinely differ; the rule governs a value a caller branches on, not an element type consumed by folding.
- Preserve reasoning verbatim across turns: re-emit every trace in place, unconditionally. Trimming is the application's job.
- Honor user inputs faithfully, even invalid ones, and make no promise about how a provider will respond: never probe an endpoint to learn its errors, never add client-side guards guessing at provider-side rules, and do not restate this per case. Reserve client-side raises for documented provider facts and for defects that would otherwise produce a silently wrong result.
- Use pydantic only where serde plus validation pay for themselves; everything else is a frozen dataclass or NamedTuple, and each qualifying model's docstring states what its validation buys. Derive every pydantic model from the checked-copy base, on which a key that is not a field is an error.
- Keep the SDKs optional dependencies the application pins directly; declare no extras. The import path is the boundary: the neutral core imports no SDK; each backend subpackage imports its SDK at module top, guarded so a missing package raises a `ModuleNotFoundError` naming what to install.
- Give each backend subpackage a constructor named for the models it selects, returning a ready `LLM`. Send a model id verbatim with no aliases, so one string appears in application code, on the wire, and in traces. Require only the model, plus `pricing` where no carried table maps to the model id; document each parameter and each cross-provider asymmetry on the function. Models outside the catalog are built directly from the re-exported concrete adapter. Prices are the one provider fact not verifiable by introspection; each subpackage docstring carries the source URL.
- Tier the public surface by audience: applications import from top-level `langchaint` and the backend subpackages; adapter authors import from `langchaint.adapter` and `langchaint.conformance`. Top-level `__all__` re-exports only the SDK-free application surface.
- Ship OTel tracing in-tree as a thin, guarded-import subpackage off the top-level `__all__`. Premises: never fake an event boundary a span measures; the mapper gets attribute names and values, never the `GenerationInput`; catch and log telemetry failures, never propagate; wrap unconditionally, and leave enable/disable/routing to OTel SDK configuration. Record message content only through `capture_message_content`, a required keyword with no default. Use a convention key wherever one exists; reserve `langchaint.*` for what the convention lacks.

## Module map

One line per module saying what it is for; the module docstring is the spec of what it holds. No symbol lists: an inventory goes stale on every added name.

- `llm.py`: the client `LLM` and the `BoundLLM` its `bind` returns.
- `adapter.py`: the neutral base contract, dual-audience; imports no SDK.
- `conformance.py`: the invariants every adapter holds to, as a test class an adapter author inherits; imports no SDK and no test runner.
- `rate_limiter.py`: retrying and pacing.
- `exceptions.py`: the error vocabulary.
- `response.py`: the generate results and their flattening to a calls table and an attempts table.
- `call.py`: the per-call history: one attempt's record, the frozen `CallRecord` every result carries, and the ledger the retry loops drive.
- `streaming.py`: the stream handle.
- `tools.py`: the tool forms, the `Tool` protocol, `ToolManager`, and the dispatch outcome types.
- `messages.py`: the provider-neutral message tree and content parts.
- `usage.py`: token accounting and the per-category costs that travel with it.
- `checked_copy.py`: the base of langchaint's pydantic models.
- `pricing.py`: the arithmetic that spends a rate and the `Billing` an attempt carries; imports no SDK and no error class. A rate table is provider-shaped and lives in the backend subpackage whose adapter spends it.
- `anthropic/`, `openai/`: the backend subpackages; importing one requires its SDK.
- `inference_params.py`: the inference parameters.
- `tracing/`: the OTel subpackage; importing it requires opentelemetry-api, and it is off the top-level `__all__`.

## Checks

Trigger: before committing. Run `scripts/CI.sh`; fix every error it reports and rerun until it reports zero.

It runs `pyrefly check`, `ruff check`, `ruff format --check`, and `pytest` through `uv run`, so the tools resolve from the locked dev group. The de-selected docstring rules and their reasons are in `pyproject.toml`. Keep the tests offline (constructed SDK objects, stub adapters, no API keys).

## Releasing

Trigger: releasing a version. Bump `version` in `pyproject.toml` and push to `main`. Never create a `v*` tag by hand: `.github/workflows/publish.yml` publishes to PyPI and cuts the tag, and its gate publishes only a version strictly above every existing `v*` tag, so a hand-made tag stops the release it was meant to mark.

Pushing the bump to `main` is therefore the release act, and PyPI does not accept a re-upload of a version that already exists. Confirm with the user before pushing a commit that changes `version`.

# Casts

`cast` is on the global code-smell list, and here its keep-with-a-comment escape does not apply: redesign every `cast` away except at two boundaries. First, a deliberately-opaque value re-enters a typed API whose own serialization produced it, and the alternative is worse (for example a revalidation that silently reshapes the payload). Second, a langchaint vocabulary is deliberately wider than the SDK literal it is sent as, under "Honor user inputs faithfully". Keep a surviving `cast` to one line, its comment naming which boundary.
