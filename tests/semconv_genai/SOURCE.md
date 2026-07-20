# Vendored GenAI payload schemas

Source: <https://github.com/open-telemetry/semantic-conventions-genai>, `model/gen-ai/`, at commit `c26a2c21d1ee70d5231bd440c7b48d3c94ee506a`.

Licensed Apache-2.0 by the OpenTelemetry Authors, copied unmodified.

These are the JSON Schemas the GenAI semantic convention attaches to the span
attributes whose semconv type is `any`, which is every attribute carrying a
structured payload. Nothing publishes them to a package registry, so the copies
here are the pin. `tests/test_tracing.py` validates against them every payload the
tracing module emits, less the paths its `_UNVALIDATED_PAYLOAD_ATTRIBUTES` exempts
and documents. What these schemas do and do not enforce is recorded there too:
each `anyOf` over element types ends in a catch-all arm, so a green run means less
than full conformance.

Refresh with `uv run python -m scripts.refresh_semconv_genai`, then read `git diff`.

| attribute | schema |
| --- | --- |
| `gen_ai.input.messages` | `gen-ai-input-messages.json` |
| `gen_ai.output.messages` | `gen-ai-output-messages.json` |
| `gen_ai.system_instructions` | `gen-ai-system-instructions.json` |
| `gen_ai.tool.call.arguments` | `gen-ai-tool-call-arguments.json` |
| `gen_ai.tool.call.result` | `gen-ai-tool-call-result.json` |
| `gen_ai.tool.definitions` | `gen-ai-tool-definitions.json` |
