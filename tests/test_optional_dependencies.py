"""Cover imports without optional dependencies."""

import subprocess
import sys
from pathlib import Path


def _run_without_numpy(source: str) -> subprocess.CompletedProcess[str]:
    """Run `source` after blocking `numpy` imports."""
    blocked_source = 'import sys\nsys.modules["numpy"] = None\n' + source
    return subprocess.run(
        [sys.executable, "-c", blocked_source],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generation_imports_without_numpy() -> None:
    """Import generation APIs without importing `numpy`."""
    completed = _run_without_numpy(
        "import langchaint\n"
        "from langchaint.openai import OpenAI\n"
        'assert langchaint.LLM.__name__ == "LLM"\n'
        'assert OpenAI.__name__ == "OpenAI"\n'
    )

    assert completed.returncode == 0, completed.stderr


def test_embedding_apis_name_the_install_for_missing_numpy() -> None:
    """Name each supported install when an embedding API lacks `numpy`."""
    completed = _run_without_numpy(
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
        "    asyncio.run(client.close())\n"
    )

    assert completed.returncode == 0, completed.stderr
