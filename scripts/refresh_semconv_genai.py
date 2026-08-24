"""Refresh vendored GenAI validation data from OpenTelemetry.

Run `uv run python -m scripts.refresh_semconv_genai`.
"""

import json
import pathlib
import re
import urllib.request

REPO = "open-telemetry/semantic-conventions-genai"
BRANCH = "main"
MODEL_DIR = "model/gen-ai"
DESTINATION = pathlib.Path(__file__).parent.parent / "tests" / "semconv_genai"
SOURCE_DOC = DESTINATION / "SOURCE.md"
REGISTRY_FILE = "registry.yaml"
PROVIDER_NAME_VALUES_FILE = "provider-name-values.json"

ATTRIBUTE_SCHEMA_FILES = {
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.definitions": "gen-ai-tool-definitions.json",
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.tool.call.arguments": "gen-ai-tool-call-arguments.json",
    "gen_ai.tool.call.result": "gen-ai-tool-call-result.json",
}
"""Map each traced payload attribute to its validation schema."""


def fetch(url: str) -> bytes:
    """Read one URL.

    Raises:
        urllib.error.HTTPError: The server returns an error status.
        urllib.error.URLError: The host cannot be reached.
    """
    with urllib.request.urlopen(url) as response:
        content: bytes = response.read()
    return content


def resolve_head_sha() -> str:
    """Resolve BRANCH to an upstream revision.

    Raises:
        urllib.error.HTTPError: The commits endpoint returns an error status.
        urllib.error.URLError: The host cannot be reached.
        json.JSONDecodeError: The response body is not JSON.
        KeyError: The response contains no sha.
    """
    payload = json.loads(fetch(f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"))
    sha: str = payload["sha"]
    return sha


def provider_name_values_from_registry(content: bytes) -> tuple[str, ...]:
    """Extract and sort gen_ai.provider.name values.

    Raises:
        ValueError: The registry attribute is missing, duplicated, empty, or malformed.
    """
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{REGISTRY_FILE} must be UTF-8") from error
    attribute_pattern = re.compile(r"^  - key: gen_ai\.provider\.name\s*$")
    starts = [index for index, line in enumerate(lines) if attribute_pattern.fullmatch(line)]
    if len(starts) != 1:
        raise ValueError(f"{REGISTRY_FILE} must define gen_ai.provider.name exactly once")
    start = starts[0]
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("  - key:")),
        len(lines),
    )
    attribute_lines = lines[start:end]
    if attribute_lines.count("    type:") != 1:
        raise ValueError("gen_ai.provider.name must define one type mapping")
    if attribute_lines.count("      members:") != 1:
        raise ValueError("gen_ai.provider.name type must define one `members` list")
    members_start = attribute_lines.index("      members:") + 1
    members_end = next(
        (
            index
            for index in range(members_start, len(attribute_lines))
            if attribute_lines[index].strip()
            and len(attribute_lines[index]) - len(attribute_lines[index].lstrip()) <= 6
        ),
        len(attribute_lines),
    )
    member_starts = [
        index
        for index in range(members_start, members_end)
        if attribute_lines[index].startswith("        - ")
    ]
    if not member_starts:
        raise ValueError("gen_ai.provider.name `members` must not be empty")
    values: list[str] = []
    for position, member_start in enumerate(member_starts):
        member_end = (
            member_starts[position + 1] if position + 1 < len(member_starts) else members_end
        )
        value_lines = [
            line.removeprefix("          value: ")
            for line in attribute_lines[member_start:member_end]
            if line.startswith("          value: ")
        ]
        if len(value_lines) != 1:
            raise ValueError("each entry in gen_ai.provider.name `members` must define one value")
        try:
            value: object = json.loads(value_lines[0])
        except json.JSONDecodeError as error:
            raise ValueError("gen_ai.provider.name values must be JSON strings") from error
        if not isinstance(value, str) or not value:
            raise ValueError("gen_ai.provider.name values must be non-empty strings")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("gen_ai.provider.name values must be unique")
    return tuple(sorted(values))


def render_source_doc() -> str:
    """Render SOURCE_DOC with source paths and refresh instructions."""
    lines = [
        "# Vendored GenAI semantic-convention data",
        "",
        f"Source repository: <https://github.com/{REPO}>.",
        f"Source branch: `{BRANCH}`.",
        "",
        f"Payload schemas come from `{MODEL_DIR}/gen-ai-*.json`.",
        f"Provider names come from `{MODEL_DIR}/{REGISTRY_FILE}`.",
        "",
        "OpenTelemetry Authors license these files under Apache-2.0.",
        "Payload schemas are copied unchanged.",
        "Provider names are extracted and sorted.",
        "",
        "Refresh with `uv run python -m scripts.refresh_semconv_genai`, then read `git diff`.",
        "",
        "| attribute | schema |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{key}` | `{file}` |" for key, file in sorted(ATTRIBUTE_SCHEMA_FILES.items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Fetch every vendored file from one upstream commit.

    Raises:
        urllib.error.HTTPError: An upstream request returns an error status.
        urllib.error.URLError: An upstream host cannot be reached.
        json.JSONDecodeError: The commits endpoint response is not JSON.
        KeyError: The commits endpoint response contains no sha.
        ValueError: `registry.yaml` is invalid.
        OSError: A vendored path cannot be created or written.
    """
    sha = resolve_head_sha()
    schema_contents = {
        file: fetch(f"https://raw.githubusercontent.com/{REPO}/{sha}/{MODEL_DIR}/{file}")
        for file in sorted(ATTRIBUTE_SCHEMA_FILES.values())
    }
    registry = fetch(f"https://raw.githubusercontent.com/{REPO}/{sha}/{MODEL_DIR}/{REGISTRY_FILE}")
    provider_name_values = provider_name_values_from_registry(registry)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for file, content in schema_contents.items():
        _ = (DESTINATION / file).write_bytes(content)
        print(f"wrote {DESTINATION / file} ({len(content)} bytes)")
    provider_name_values_path = DESTINATION / PROVIDER_NAME_VALUES_FILE
    _ = provider_name_values_path.write_text(json.dumps(provider_name_values, indent=2) + "\n")
    print(f"wrote {provider_name_values_path}")
    _ = SOURCE_DOC.write_text(render_source_doc())
    print(f"wrote {SOURCE_DOC} at {sha}")


if __name__ == "__main__":
    main()
