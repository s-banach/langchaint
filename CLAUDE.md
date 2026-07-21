# langchaint

A provider-neutral LLM client library. Alpha: the API is unstable and may change without notice.

## Docstrings and comments

Write docstrings and comments self-contained for human readers: state the constraint or reasoning inline; never cite internal dev documents, decision logs, spec files, or dead alternatives to the live code.
Never refer to the repo's own prior state. Diff-relative prose is a sentence that only makes sense to a reader who saw the change that introduced it: wording like "as before", "no longer", or "now", a reference to a state that exists nowhere in the current tree, or a justification for a question no reader of the file would ask. The test: a sound sentence reads the same whether the code was born in its current shape or arrived there by refactor.

Module docstrings are the spec of record for mechanics; CLAUDE.md is the spec of record for principles. Write in CLAUDE.md only a cross-module rule, its criterion, and at most one edge-case example per rule; keep how a behavior works in the docstring of the code that implements it, and when such a behavior changes, update that docstring, not CLAUDE.md. The durability test: a CLAUDE.md sentence reads the same after any refactor that preserves the design tenets, so a sentence naming a symbol belongs in that symbol's docstring, except where the rule is about that name (`input_tokens`, `StreamItem`).
Record a rejected alternative and its one-line reason where its re-litigation would start: in the docstring of the code the objection targets, or, when the objection targets a deliberate absence (there is no code to object to), as one line in CLAUDE.md beside the principle the absence instantiates. The reviewer's design objections (rule 8 in `.claude/agents/commit-reviewer.md`) honor a recorded rejection in either home. When condensing CLAUDE.md, move a rejection to the docstring it belongs in; never sweep one into deletion.

Document what a function raises. In every function whose body can raise (directly, by re-raising, or by propagating a documented raise from a helper), name the exception types a caller may see and the condition for each, in a `Raises:` section when the raise is direct or in prose when it is not. Enforce this in review, not lint: the de-selected ruff docstring rules cannot catch a missing or stale raise (see the note in `pyproject.toml`).

## Provider facts

Never assert provider behavior (wire parameters, usage-field semantics, exception taxonomies, cache rules) from memory; verify it against the installed `anthropic`/`openai` packages by introspection before writing or reviewing code that depends on it.

Put a verified fact in the docstring of the code that depends on it, with the SDK version when the fact could drift. Record in CLAUDE.md only facts that justify a rule spanning modules, cited inside that rule.

## Naming rules

- Make every name explicit: `system_prompt` not `system`, `inference_params` not `params`.
- Give the keyword and the variable passed to it one name: `tool_manager=tool_manager`, because `name_1=name_2` gives one concept two names.
- Use one name per concept end to end; no aliases. Call the project "langchaint", never "the package" or "the library"; use "package" only for its Python meanings.
- Say "adapter" for an implementation of the class `Adapter`, including in compounds, and "provider" for anthropic and openai themselves and for a platform serving their models; a concrete name composes the two (`AnthropicMessagesAdapter`). "Provider" is wider than the company because the serving platform counts: one adapter reports a different `provider_name` over a direct client than over a Bedrock one.
- Prefer neutral over provider vocabulary: when providers disagree, take the majority wire name or a neutral one (`ToolCall` not `ToolUse`).
- Never write bare `input_tokens`: anthropic's field of that name excludes cache reads while openai's equivalent includes them. Use the partition `input_tokens_cache_read`/`_cache_write`/`_cache_none` and the derived `input_tokens_total`.
- Put units and encodings in names (`cost_in_usd`, `elapsed_seconds`); mark unparsed JSON text with the `_json` suffix.
- Use family prefixes to keep related fields adjacent in sorts and completions: `input_tokens_*`, `generate_one`/`generate_many` (arity in the suffix).
- Never stutter with the holder (`tool.name`, not `tool.tool_name`); carry the full name on cross-object references (`tool_call_id` on `ToolMessage`).
- Give the plain noun to the interface, because the protocol name is read far more often than any concrete name is written; name concrete forms by the technology their argument spec is written in, except a form distinguished by its fixed behavior, which is named for that behavior.
- Use `cache_breakpoint` as the neutral name for a user-placed prompt-cache boundary: True on a part means the reusable prompt prefix ends there.
- Count `max_attempts` as requests sent including the first, so 1 means no retrying; configure the SDK client so it never retries beneath langchaint and the count stays true.
- Keep `content`, `output`, and `reasoning` three concepts, never one word: a model-facing message body, the generation result payload, and provider round-trip data replayed opaquely.

## Design rules

- Generate only via binding: `bind` freezes everything that determines the cacheable prompt prefix; changing parameters is `rebind`, never a per-call override.
- Never choose a billing-relevant configuration for the user: `automatic_prompt_caching` is a required keyword with no default (an unstated `False` is a billing choice as real as opting in), and any convenience on top of user-stated caching is opt-in and default-off. Honor user-placed `cache_breakpoint` marks under either binding value. A per-part TTL is rejected until someone asks.
- Leave the tool loop to the application: ship no agent loop, and make a tool function return data, never a control-flow signal. No concurrency-limit parameter; no app-supplied validator parameter (a rule the schema cannot express lives in the tool function).
- Keep one `RateLimiter` owning retrying and pacing: one instance is one shared budget for the account it guards, gating every request-start path. No `requests_per_minute`: an in-flight bound self-adjusts throughput.
- Wrap official SDK clients and delegate stream assembly and structured-output parsing to the SDK. Write no wire TypedDicts by hand.
- Split the error taxonomy along two orthogonal axes: retry (retry transient, propagate the rest; "non-retriable" is a concept, not a class) and batch (abort-batch cancels siblings; a generation error is one item's failure row). Classify a parse that returns no output as refusal, truncation, or transient, never silently-wrong data.
- Admit a field to `Usage` only if it is a provider-invariant counter or the priced scalar absorbing provider-variant billing structure; keep provider-specific detail on the raw SDK usage beside it, and never let `usage` be `None` on a carrier.
- Compute the cost breakdown on demand, never stored (a derived cost does not fold like a counter), routed through the same pricing call as the stored scalar so they cannot drift.
- Scope `usage` as the paid total across every attempt, on success and on failure, folded from the attempt records, the one source of truth.
- No separate Bedrock adapter: the existing adapters take the SDKs' bundled Bedrock clients, and there is no Converse adapter. Never fabricate a price or a model catalog: default pricing only from a carried rate table, require it where no table maps to the model id, and call a first-party list price on a rate-setting platform an estimate.
- Support OpenAI through the Responses API only; no Chat Completions adapter, and third-party compatible servers are out of scope.
- Yield from streams only `StreamItem = str | ToolCall`: no delta items, no usage or stop items (those live on the final response).
- Never take data out of an SDK object only to reconstruct it in a langchaint object of the same shape; pass provider values through by reference, constructing a langchaint model only where the shape genuinely changes.
- Discriminate outcomes by type, not a nullable flag: return a union of frozen dataclasses, one arm per outcome, extra data as required non-optional fields, so matching narrows and no consumer writes `cast` or `assert x is not None`. Split only where fields genuinely differ; the rule governs a value a caller branches on, not an element type consumed by folding.
- Preserve reasoning verbatim across turns: re-emit every trace in place, unconditionally. Trimming is the app's job by rebuilding concluded turns; no bind-time on/off parameter.
- Honor user inputs faithfully, even invalid ones, and make no promise about how a provider will respond: never probe an endpoint to learn its errors, never add client-side guards guessing at provider-side rules, and do not restate this per case. Reserve client-side raises for documented provider facts and for defects that would otherwise produce a silently wrong result.
- Use pydantic only where serde plus validation pay for themselves; everything else is a frozen dataclass or NamedTuple, and each qualifying model's docstring states what its validation buys. Derive every pydantic model from the checked-copy base, on which a key that is not a field is an error.
- Keep the SDKs optional dependencies the application pins directly; declare no extras. The import path is the boundary: the neutral core imports no SDK; each backend subpackage imports its SDK at module top, guarded so a missing package raises a `ModuleNotFoundError` naming what to install.
- Give each backend subpackage a constructor named for the models it selects, returning a ready `LLM`. Send a model id verbatim with no aliases, so one string appears in application code, on the wire, and in traces. Require only the model, plus `pricing` where no carried table maps to the model id; document each parameter and each cross-provider asymmetry on the function. Models outside the catalog are built directly from the re-exported concrete adapter. Prices are the one provider fact not verifiable by introspection; each subpackage docstring carries the source URL.
- Tier the public surface by audience: applications import from top-level `langchaint` and the backend subpackages; adapter authors import from `langchaint.adapter`. Top-level `__all__` re-exports only the SDK-free application surface.
- Keep the inference parameters deliberately minimal, None leaving the provider default in place; reach an unmapped provider parameter by subclassing the concrete adapter, never a passthrough dict.
- Ship OTel tracing in-tree as a thin, guarded-import subpackage off the top-level `__all__`. Premises: never fake an event boundary a span measures; the mapper gets attribute names and values, never the conversation; catch and log telemetry failures, never propagate; wrap unconditionally, and leave enable/disable/routing to OTel SDK configuration. Record conversation content only through `capture_message_content`, a required keyword with no default. Use a convention key wherever one exists; reserve `langchaint.*` for what the convention lacks.
- Verified provider facts that cross module boundaries (checked against anthropic 0.116.0 / openai 2.45.0): both SDK clients retry internally by default (`DEFAULT_MAX_RETRIES = 2`) and honor `retry-after-ms`/`retry-after` only up to 60 seconds; both SDKs' `APIStatusError` exposes `.response` (an httpx.Response) whose headers carry retry-after; `with_options(max_retries=0)` returns a client copy on all five client classes.

## Module map

One line per module saying what it is for; the module docstring is the spec of what it holds. No symbol lists: an inventory goes stale on every added name.

- `llm.py`: the client `LLM` and the `BoundLLM` its `bind` returns.
- `adapter.py`: the neutral base contract, dual-audience; imports no SDK.
- `rate_limiter.py`: retrying and pacing.
- `exceptions.py`: the error vocabulary.
- `response.py`: the generate results and their flattening to one row shape.
- `streaming.py`: the stream handle.
- `tools.py`: the tool forms, the `Tool` protocol, `ToolManager`, and the dispatch outcome types.
- `messages.py`: the provider-neutral message tree and content parts.
- `usage.py`: token accounting and the `cost_in_usd` that travels with it.
- `checked_copy.py`: the base of langchaint's pydantic models.
- `pricing.py`: the neutral cost arithmetic; imports no SDK and no error class.
- `anthropic/`, `openai/`: the backend subpackages; importing one requires its SDK.
- `inference_params.py`: the inference parameters.
- `tracing/`: the OTel subpackage; importing it requires opentelemetry-api, and it is off the top-level `__all__`.

## Checks

Trigger: before committing. Run `scripts/CI.sh`; fix every error it reports and rerun until it reports zero.

It runs `pyrefly check`, `ruff check`, and `pytest` through `uv run`, so the tools resolve from the locked dev group. The de-selected docstring rules and their reasons are in `pyproject.toml`. Keep the tests offline (constructed SDK objects, stub adapters, no API keys). `tests/*` carries a `SLF001` per-file-ignore because the tests exercise private helpers directly.

# Commit Review

After each commit lands, spawn the `commit-reviewer` agent (`.claude/agents/commit-reviewer.md`) with the commit sha as its prompt. Before that spawn, `scripts/CI.sh` reports zero errors and you have run the author's pre-commit pass ("Re-read before committing" in the global CLAUDE.md) to completion on every staged file. The reviewer is the second pass, which is what lets its rule 2 forbid re-running the checks. Scale the reviewer model with the commit's stakes: Opus for serious work, Sonnet only for the most trivial docstring changes.

Done gate: count a commit as Done only after its review has returned, confirmed issues are fixed, and every objection is resolved: adopted by editing what its premise challenges, or rejected by recording the alternative and reason per the placement rule in Docstrings and comments. Deferring an objection is not resolving it. Keep one feature one commit: amend fixes into the reviewed commit and re-review it; the resolution edit rides in the same amend, and there is no separate open-items file. Never hand off or build on a not-Done commit; the loop's own amend-and-force-push is how a commit reaches Done, not a violation.

# Code smells

Trigger: you write one of the constructs below. Redesign to remove it. If every redesign you find is worse, keep the construct with a comment stating why. Stop when the construct is gone or carries that comment.

- `**kwargs` unpacking
- `Any` for typing
- `cast`. The keep-with-a-comment escape does not apply: redesign every `cast` away except at two boundaries. First, a deliberately-opaque value re-enters a typed API whose own serialization produced it, and the alternative is worse (for example a revalidation that silently reshapes the payload). Second, a langchaint vocabulary is deliberately wider than the SDK literal it is sent as, under "Honor user inputs faithfully". Keep a surviving `cast` to one line, its comment naming which boundary.
- Unpacking data from one container (list, dict, pydantic object) and back into another.
