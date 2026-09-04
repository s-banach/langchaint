"""Cover imports without optional dependencies."""

import subprocess
import sys
from pathlib import Path


def _run_with_blocked_import(module_name: str, source: str) -> subprocess.CompletedProcess[str]:
    """Run `source` after blocking imports of `module_name`."""
    blocked_source = f"import sys\nsys.modules[{module_name!r}] = None\n" + source
    return subprocess.run(
        [sys.executable, "-c", blocked_source],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generation_imports_without_numpy() -> None:
    """Import generation APIs without importing `numpy`."""
    completed = _run_with_blocked_import(
        "numpy",
        "import langchaint\nfrom langchaint.openai import OpenAI\n",
    )

    assert completed.returncode == 0, completed.stderr


def test_span_parsing_imports_without_opentelemetry() -> None:
    """Run `parse_otel` when `opentelemetry` is unavailable."""
    completed = _run_with_blocked_import(
        "opentelemetry",
        "from langchaint.span_parsing import parse_otel\n"
        'parsed = parse_otel({"gen_ai.operation.name": "chat"})\n'
        'assert parsed.operation_name == "chat"\n',
    )

    assert completed.returncode == 0, completed.stderr


def test_normalized_error_validation_imports_no_provider_backend() -> None:
    """Validating normalized error JSON imports no provider backend."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from pydantic import TypeAdapter\n"
            "from langchaint import GenerationErrorRecord\n"
            'payload = b\'{"call":{"model":"m","provider_name":"p","attempt_records":[],"elapsed_seconds":0.0},"error_text":"bad","kind":"invalid_request_error"}\'\n'
            "record = TypeAdapter(GenerationErrorRecord).validate_json(payload)\n"
            'assert record.kind == "invalid_request_error"\n'
            'provider_prefixes = ("langchaint.anthropic", "langchaint.cohere", "langchaint.deepseek", "langchaint.gemini", "langchaint.openai")\n'
            "assert not any(name.startswith(provider_prefixes) for name in sys.modules)\n",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_embedding_apis_name_the_install_for_missing_numpy() -> None:
    """Name each supported install when an embedding API lacks `numpy`."""
    completed = _run_with_blocked_import(
        "numpy",
        "import asyncio\n"
        "import langchaint\n"
        "from openai import AsyncOpenAI\n"
        "from langchaint.openai import OpenAI\n"
        'neutral_expected = "langchaint embeddings require the numpy package; install numpy."\n'
        'openai_expected = "OpenAI embeddings require numpy and tiktoken; install langchaint[openai-embedding]."\n'
        "try:\n"
        "    _ = langchaint.EmbeddingModel\n"
        "except ModuleNotFoundError as error:\n"
        "    assert str(error) == neutral_expected\n"
        "else:\n"
        '    raise AssertionError("EmbeddingModel did not require numpy")\n'
        'client = AsyncOpenAI(api_key="offline")\n'
        "openai = OpenAI(client=client)\n"
        "try:\n"
        '    _ = openai.embedding_model("text-embedding-3-small")\n'
        "except ModuleNotFoundError as error:\n"
        "    assert str(error) == openai_expected\n"
        "else:\n"
        '    raise AssertionError("embedding_model did not require numpy")\n'
        "finally:\n"
        "    asyncio.run(client.close())\n",
    )

    assert completed.returncode == 0, completed.stderr
