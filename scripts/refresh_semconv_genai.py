"""Refresh committed GenAI semantic-convention data through Weaver.

Run `uv run python -m scripts.refresh_semconv_genai`.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).parent.parent
SOURCE_REPOSITORY = "open-telemetry/semantic-conventions-genai"
SOURCE_URL = f"https://github.com/{SOURCE_REPOSITORY}.git"
SOURCE_REFERENCE = "main"
SOURCE_CHECKOUT = ROOT / "build" / "semantic-conventions-genai-source"
PREPARED_SOURCE_REF = "refs/langchaint/prepared"
SOURCE_MODEL_DIRECTORY = SOURCE_CHECKOUT / "model"
DESTINATION = ROOT / "tests" / "semconv_genai"
SOURCE_DOC = DESTINATION / "SOURCE.md"
TEMPLATES = ROOT / "scripts" / "semconv_genai_templates"
GENERATED_ATTRIBUTES_FILE = "chat-span-attributes.json"
WEAVER_TARGET = "chat-span"
OBSOLETE_FILES = {"provider-name-values.json"}

ATTRIBUTE_SCHEMA_FILES = {
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.definitions": "gen-ai-tool-definitions.json",
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.tool.call.arguments": "gen-ai-tool-call-arguments.json",
    "gen_ai.tool.call.result": "gen-ai-tool-call-result.json",
}
"""Map each structured attribute to its committed JSON schema."""


def _run(command: list[str], *, working_directory: pathlib.Path = ROOT) -> str:
    completed = subprocess.run(
        command,
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolved_main_sha() -> str:
    output = _run(["git", "ls-remote", SOURCE_URL, f"refs/heads/{SOURCE_REFERENCE}"])
    fields = output.split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{SOURCE_REFERENCE}":
        raise ValueError(f"git ls-remote returned malformed {SOURCE_REFERENCE!r} output")
    sha = fields[0]
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise ValueError(f"git ls-remote returned malformed commit SHA {sha!r}")
    return sha


def _prepare_source_checkout(resolved_sha: str) -> None:
    SOURCE_CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
    if not (SOURCE_CHECKOUT / ".git").is_dir():
        if SOURCE_CHECKOUT.exists():
            raise ValueError(f"{SOURCE_CHECKOUT} exists without a Git checkout")
        _ = _run(["git", "clone", SOURCE_URL, str(SOURCE_CHECKOUT)])
    origin_url = _run(["git", "remote", "get-url", "origin"], working_directory=SOURCE_CHECKOUT)
    if origin_url.rstrip("/").removesuffix(".git") != SOURCE_URL.removesuffix(".git"):
        raise ValueError(f"{SOURCE_CHECKOUT} origin is {origin_url!r}, expected {SOURCE_URL!r}")
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        working_directory=SOURCE_CHECKOUT,
    )
    if status:
        raise ValueError(f"{SOURCE_CHECKOUT} contains uncommitted files")
    current_sha = _run(["git", "rev-parse", "HEAD"], working_directory=SOURCE_CHECKOUT)
    if current_sha != resolved_sha:
        _ = _run(["git", "fetch", "origin", resolved_sha], working_directory=SOURCE_CHECKOUT)
        _ = _run(["git", "checkout", "--detach", resolved_sha], working_directory=SOURCE_CHECKOUT)
        current_sha = _run(["git", "rev-parse", "HEAD"], working_directory=SOURCE_CHECKOUT)
        if current_sha != resolved_sha:
            raise ValueError(f"{SOURCE_CHECKOUT} is at {current_sha}, expected {resolved_sha}")


def _resolved_source_sha() -> str:
    if (SOURCE_CHECKOUT / ".git").is_dir():
        try:
            return _run(
                ["git", "rev-parse", "--verify", PREPARED_SOURCE_REF],
                working_directory=SOURCE_CHECKOUT,
            )
        except subprocess.CalledProcessError:
            pass
    return _resolved_main_sha()


def _required_weaver_version() -> str:
    versions_path = SOURCE_CHECKOUT / "versions.env"
    matches = [
        line.removeprefix("WEAVER_VERSION=")
        for line in versions_path.read_text().splitlines()
        if line.startswith("WEAVER_VERSION=")
    ]
    if len(matches) != 1 or not matches[0].startswith("v"):
        raise ValueError(f"{versions_path} must define one WEAVER_VERSION")
    return matches[0]


def _weaver_executable(required_version: str) -> str:
    executable = shutil.which("weaver")
    if executable is None:
        raise ValueError(
            f"install Weaver {required_version} from https://github.com/open-telemetry/weaver/releases"
        )
    version_output = _run([executable, "--version"])
    expected_output = f"weaver {required_version.removeprefix('v')}"
    if version_output != expected_output:
        raise ValueError(
            f"{executable} reports {version_output!r}; install Weaver {required_version}"
        )
    return executable


def _validate_json_file(path: pathlib.Path) -> None:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")


def _render_source_doc(resolved_sha: str, weaver_version: str) -> str:
    schema_paths = "\n".join(
        f"- `model/gen-ai/{file}` for `{attribute}`."
        for attribute, file in sorted(ATTRIBUTE_SCHEMA_FILES.items())
    )
    return f"""# Committed GenAI semantic-convention data

Source repository: <https://github.com/{SOURCE_REPOSITORY}>.
Tracked reference: `{SOURCE_REFERENCE}`.
Resolved commit SHA: `{resolved_sha}`.
Resolved Weaver version: `{weaver_version}`.
License: Apache-2.0.

`{GENERATED_ATTRIBUTES_FILE}` is generated from the resolved `gen_ai.inference.client` span and its provider refinements in `model/gen-ai/spans.yaml` and `model/gen-ai/registry.yaml`.
Weaver also resolves the core registry dependency declared by `model/manifest.yaml`.

Structured schemas are copied from these source paths:

{schema_paths}

Refresh with `uv run python -m scripts.refresh_semconv_genai`.
"""


def _generate_staged_files(
    temporary_directory: pathlib.Path,
    weaver_executable: str,
    resolved_sha: str,
    weaver_version: str,
) -> pathlib.Path:
    generated_directory = temporary_directory / "generated"
    staged_directory = temporary_directory / "staged"
    staged_directory.mkdir()
    _ = _run([
        weaver_executable,
        "registry",
        "generate",
        WEAVER_TARGET,
        str(generated_directory),
        "--templates",
        str(TEMPLATES),
        "--registry",
        str(SOURCE_MODEL_DIRECTORY),
        "--v2",
    ])
    generated_attributes = generated_directory / GENERATED_ATTRIBUTES_FILE
    _validate_json_file(generated_attributes)
    _ = shutil.copyfile(generated_attributes, staged_directory / GENERATED_ATTRIBUTES_FILE)
    for schema_file in sorted(ATTRIBUTE_SCHEMA_FILES.values()):
        source_path = SOURCE_MODEL_DIRECTORY / "gen-ai" / schema_file
        _validate_json_file(source_path)
        _ = shutil.copyfile(source_path, staged_directory / schema_file)
    _ = (staged_directory / SOURCE_DOC.name).write_text(
        _render_source_doc(resolved_sha, weaver_version)
    )
    return staged_directory


def _replace_committed_files(staged_directory: pathlib.Path) -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for obsolete_file in OBSOLETE_FILES:
        obsolete_path = DESTINATION / obsolete_file
        if obsolete_path.exists():
            obsolete_path.unlink()
    for staged_path in sorted(staged_directory.iterdir()):
        destination_path = DESTINATION / staged_path.name
        _ = staged_path.replace(destination_path)
        print(f"wrote {destination_path}")


def main() -> None:
    """Refresh committed data from one resolved source commit with its required Weaver."""
    resolved_sha = _resolved_source_sha()
    _prepare_source_checkout(resolved_sha)
    weaver_version = _required_weaver_version()
    weaver_executable = _weaver_executable(weaver_version)
    print(f"reference: {SOURCE_REFERENCE}")
    print(f"commit: {resolved_sha}")
    print(f"weaver: {weaver_version}")
    with tempfile.TemporaryDirectory(prefix="semconv-genai-", dir=ROOT / "build") as directory:
        staged_directory = _generate_staged_files(
            pathlib.Path(directory), weaver_executable, resolved_sha, weaver_version
        )
        _replace_committed_files(staged_directory)


if __name__ == "__main__":
    main()
