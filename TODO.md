# TODO

Work found by comparing langchaint against the 24 example programs in the pydantic-ai repository.
Each item states what to do and how to know it is done.

## A response carrying a built-in tool call produces a truncated AssistantMessage

All four adapters drop the provider's own built-in tool blocks, and nothing raises.

- `anthropic/messages_adapter.py`, `_assistant_message_from`: a `server_tool_use` block and its result block are dropped.
- `openai/responses_adapter.py`, `_assistant_message_from`: an output item that is not `reasoning`, `function_call`, or `message` is dropped.
- `openai/chat_completions_adapter.py`, `_assistant_message_from`: `message.annotations` is never read.
- `gemini/generate_content_adapter.py`, `_assistant_message_from`: a part carrying `executable_code` or `code_execution_result` is dropped.

The streaming path drops the same blocks.

One provider tool is already reachable, and only on the Chat Completions adapter.
`web_search_options` is a Chat Completions request key that adapter does not populate, so `extra_body={"web_search_options": {...}}` binds and sends, and the citations openai returns on `message.annotations` are dropped.
Everywhere else the entry would have to go in the wire `tools` array, which every adapter refuses as a key it populates.
`example-provider-tools.md` sketches the binding parameter that would open the other three.

Decide what an arriving built-in tool block does, then apply that decision to all four adapters.
Stop when a constructed response holding such a block reaches the chosen behavior under test, in all four adapters, on both the generate path and the stream path.

## `bind` does not say what a caller writes for a union or a bare list

`bind(response_format=...)` takes one `type[BaseModel]`, and langchaint synthesizes no envelope around it.
The caller writes the wrapper model.

Add that sentence to the `bind` docstring, with the two shapes it covers:
a union of outcomes becomes one model holding a `kind`-tagged field, and `list[Whale]` becomes `class WhaleList(BaseModel): response: list[Whale]`, read through `.response`.

Stop when a reader of `bind` stops looking for a union parameter.

## Two capabilities langchaint has no binding for

State the answer for each where a reader reaches for it, so a reader stops looking.

- Embeddings. The pydantic-ai RAG example needs them, and langchaint generates text only.
- Media beyond images. `ImagePart` carries bytes and a media type. There is no audio part and no image named by URL; `plan-richer-content-blocks.md` holds the support matrix and the shapes for both. Documents stay out, and `Part`'s docstring already states the conversion the application performs instead.
