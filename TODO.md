# TODO

Design questions raised in discussion and deferred, each with the facts it was raised from.
A commit-review objection does not belong here: those are resolved before their commit is Done.

## Carry the SDK response on the rejecting outcome arms

`AdapterResult` carries `raw`, the SDK response object, by reference.
The three arms an adapter reports a rejected 200 on (`Refused`, `Truncated`, `Unparsed` in `adapter.py`) carry `usage` and `usage_raw` only.
So the provider payload is dropped before the retry loop builds the error: the partial content the model generated, and the provider's own stop reason string, reach no caller.
The stop reason matters because a normalized `"other"` names nothing the caller can act on, while the SDK field beneath it holds the provider's exact word.

Proposal: give the three arms a required `raw: BaseModel` field, held by reference the way `AdapterResult.raw` is.

Cost: a required field on three arms, both adapters' construction sites for each, and the two retry loops that match the arms (`llm.py`, `streaming.py`).

Open: whether `RefusalError` and `MaxCompletionTokensExceededError` then expose it too.
Both are built from a `CallRecord` alone today, and the record holds `usage_raw` per attempt but no response object.

## A structured response cut mid-JSON never reaches the Truncated arm

Both SDKs validate the response text with no guard around it:
anthropic's `parse_text` calls `TypeAdapter.validate_json` on every text block (`anthropic/lib/_parse/_response.py`, anthropic 0.120.0),
and openai's calls `model_parse_json` on every `output_text` item (`openai/lib/_parsing/_responses.py`, openai 2.45.0).
A response that stopped mid-object therefore raises `pydantic_core.ValidationError` inside `parse`, and the exception leaves `send()` before either adapter's rejecting arms are chosen.
Both `classify` implementations return `"unrecognized"` for it, since it is no `APIStatusError`, so the item fails with `UnrecognizedError`.

Effect: a truncated structured response is reported as an error langchaint could not name, rather than as `MaxCompletionTokensExceededError`.
The item is not retried, so nothing is billed twice.
A text block that is present and unparsable raises, so each adapter's `Truncated` arm is left reachable only by a turn that carried no text at all, cut off before its text began.

Open: whether an adapter catches `ValidationError` around the SDK's parse call and reads the stop reason itself.
The obstacle is that the exception references no response object, so the stop reason is not readable from what the catch receives,
and reaching the response another way means parsing the structured output outside the SDK, against the design rule that delegates it.
