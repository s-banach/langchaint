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
