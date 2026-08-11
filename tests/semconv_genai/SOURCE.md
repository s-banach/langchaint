# Vendored GenAI semantic-convention data

Source repository: <https://github.com/open-telemetry/semantic-conventions-genai>.
Source branch: `main`.

Payload schemas come from `model/gen-ai/gen-ai-*.json`.
Provider names come from `model/gen-ai/registry.yaml`.

OpenTelemetry Authors license these files under Apache-2.0.
Payload schemas are copied unchanged.
Provider names are extracted and sorted.

Refresh with `uv run python -m scripts.refresh_semconv_genai`, then read `git diff`.

| attribute | schema |
| --- | --- |
| `gen_ai.input.messages` | `gen-ai-input-messages.json` |
| `gen_ai.output.messages` | `gen-ai-output-messages.json` |
| `gen_ai.system_instructions` | `gen-ai-system-instructions.json` |
| `gen_ai.tool.call.arguments` | `gen-ai-tool-call-arguments.json` |
| `gen_ai.tool.call.result` | `gen-ai-tool-call-result.json` |
| `gen_ai.tool.definitions` | `gen-ai-tool-definitions.json` |
