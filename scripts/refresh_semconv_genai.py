"""Refresh the vendored GenAI payload JSON schemas and record the commit they came from.

The schemas describe the structured payloads the tracing module writes into
gen_ai.system_instructions, gen_ai.tool.definitions, gen_ai.input.messages,
gen_ai.output.messages, gen_ai.tool.call.arguments, and gen_ai.tool.call.result.
They are vendored because nothing publishes them: open-telemetry/semantic-conventions-genai
releases only GitHub Releases with no package-registry artifact, and the per-language
semantic-convention packages carry attribute name constants only, no payload schemas
(opentelemetry-semantic-conventions 0.64b0 ships .py files and no data files at all).
The convention references each schema from an annotation on the attribute it describes,
where the attribute's own semconv type is `any`:

    type: any
    annotations:
      type:
        json_schema: model/gen-ai/gen-ai-tool-definitions.json

Only the six schemas the tracing module can emit are vendored; gen-ai-memory-records.json
and gen-ai-retrieval-documents.json describe attributes langchaint never writes.

Run it with zero arguments: `uv run python -m scripts.refresh_semconv_genai`. The upstream
repository, the file list, and the destination are the constants below, so the run is
committed, not assembled at the invocation. DESTINATION is resolved from this file rather
than the working directory, so a run from elsewhere refreshes the vendored copies instead
of quietly creating a tests/semconv_genai beside wherever it was launched.
It resolves the head of BRANCH first and fetches every file at that one commit,
so the schemas and the sha in SOURCE_DOC cannot disagree.
Rerunning is how drift is detected: refresh, then read `git diff` over the .json files. No diff
there means the schemas have not moved, whatever SOURCE_DOC's sha now says, because that sha
advances on every upstream commit including the many that touch no schema. A failing `scripts/CI.sh`
over a diff in the .json files names the payload that no longer conforms; a passing one means every
payload still validates, which is weaker than full conformance because each anyOf over element types
ends in a catch-all arm that admits a renamed type (the limits are in _validate_payload_attributes).
The monthly .github/workflows/refresh_semconv_genai.yml runs exactly this, applies that same
schemas-only test, and opens a pull request when they moved, so the diff reaches review without
anyone remembering to look.
"""

import json
import pathlib
import urllib.request

REPO = "open-telemetry/semantic-conventions-genai"
BRANCH = "main"
MODEL_DIR = "model/gen-ai"
DESTINATION = pathlib.Path(__file__).parent.parent / "tests" / "semconv_genai"
SOURCE_DOC = DESTINATION / "SOURCE.md"

ATTRIBUTE_SCHEMA_FILES = {
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.definitions": "gen-ai-tool-definitions.json",
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.tool.call.arguments": "gen-ai-tool-call-arguments.json",
    "gen_ai.tool.call.result": "gen-ai-tool-call-result.json",
}
"""Each payload attribute the tracing module emits, mapped to the schema describing it.

This is the fetch list and the table written into SOURCE_DOC, so a schema can only be
vendored by being named here beside the attribute that justifies it.
tests/test_tracing.py imports this map rather than restating it, and checks it against
three sources that do not come from here: the schema files on disk, the attribute keys the
tracing module emits, and upstream's own naming, which names each schema after the attribute
it describes. So an edit here that mislabels SOURCE_DOC's attribute column fails that test.
"""


def fetch(url: str) -> bytes:
    """Read one URL, raising on any non-success status.

    Raises:
        urllib.error.HTTPError: the server answered with an error status.
        urllib.error.URLError: the host could not be reached.
    """
    with urllib.request.urlopen(url) as response:
        return response.read()


def resolve_head_sha() -> str:
    """Return the full commit sha at the head of BRANCH in the upstream repository.

    Raises:
        urllib.error.HTTPError: the commits endpoint answered with an error status.
        urllib.error.URLError: the host could not be reached.
        json.JSONDecodeError: the response body was not JSON.
        KeyError: the response carried no sha, meaning the API shape changed.
    """
    payload = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"))
    return payload["sha"]


def render_source_doc(sha: str) -> str:
    """Render SOURCE_DOC recording where the schemas came from and how to refresh them."""
    lines = [
        "# Vendored GenAI payload schemas",
        "",
        f"Source: <https://github.com/{REPO}>, `{MODEL_DIR}/`, at commit `{sha}`.",
        "",
        "Licensed Apache-2.0 by the OpenTelemetry Authors, copied unmodified.",
        "",
        "These are the JSON Schemas the GenAI semantic convention attaches to the span",
        "attributes whose semconv type is `any`, which is every attribute carrying a",
        "structured payload. Nothing publishes them to a package registry, so the copies",
        "here are the pin. `tests/test_tracing.py` validates against them every payload the",
        "tracing module emits, less the paths its `_UNVALIDATED_PAYLOAD_ATTRIBUTES` exempts",
        "and documents. What these schemas do and do not enforce is recorded there too:",
        "each `anyOf` over element types ends in a catch-all arm, so a green run means less",
        "than full conformance.",
        "",
        "Refresh with `uv run python -m scripts.refresh_semconv_genai`, then read `git diff`.",
        "",
        "| attribute | schema |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{key}` | `{file}` |" for key, file in sorted(ATTRIBUTE_SCHEMA_FILES.items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Fetch every schema at one upstream commit and rewrite SOURCE_DOC with that sha.

    Raises:
        urllib.error.HTTPError: an upstream request answered with an error status.
        urllib.error.URLError: the host could not be reached.
        json.JSONDecodeError: the commits endpoint returned a body that was not JSON.
        KeyError: the commits endpoint returned no sha, meaning the API shape changed.
        OSError: a schema or SOURCE_DOC could not be written.
    """
    sha = resolve_head_sha()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for file in sorted(ATTRIBUTE_SCHEMA_FILES.values()):
        content = fetch(f"https://raw.githubusercontent.com/{REPO}/{sha}/{MODEL_DIR}/{file}")
        (DESTINATION / file).write_bytes(content)
        print(f"wrote {DESTINATION / file} ({len(content)} bytes)")
    SOURCE_DOC.write_text(render_source_doc(sha))
    print(f"wrote {SOURCE_DOC} at {sha}")


if __name__ == "__main__":
    main()
