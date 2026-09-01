# Committed GenAI semantic-convention data

Source repository: <https://github.com/open-telemetry/semantic-conventions-genai>.
Tracked reference: `main`.
Resolved commit SHA: `ac46a5d7bfe0b0f47e8ce393e2db3a2c3042f236`.
Resolved Weaver version: `v0.25.1`.
License: Apache-2.0.

`chat-span-attributes.json` is generated from the resolved `gen_ai.inference.client` span and its provider refinements in `model/gen-ai/spans.yaml` and `model/gen-ai/registry.yaml`.
`src/langchaint/_semconv_genai_structured_attributes.json` is generated from attributes that declare `annotations.type.json_schema` in the resolved registry.
Weaver also resolves the core registry dependency declared by `model/manifest.yaml`.

Structured schemas are copied from these source paths:

- `model/gen-ai/gen-ai-input-messages.json` for `gen_ai.input.messages`.
- `model/gen-ai/gen-ai-output-messages.json` for `gen_ai.output.messages`.
- `model/gen-ai/gen-ai-system-instructions.json` for `gen_ai.system_instructions`.
- `model/gen-ai/gen-ai-tool-call-arguments.json` for `gen_ai.tool.call.arguments`.
- `model/gen-ai/gen-ai-tool-call-result.json` for `gen_ai.tool.call.result`.
- `model/gen-ai/gen-ai-tool-definitions.json` for `gen_ai.tool.definitions`.

Refresh with `uv run python -m scripts.refresh_semconv_genai`.
