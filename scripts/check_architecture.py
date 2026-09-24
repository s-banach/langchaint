"""Check explicit source imports without executing langchaint.

The dependency and cycle checks include conditional and function-local imports.
The load-time check follows the imports that run when a module loads.
A load-time `from module import name` also counts every import in that module's `__getattr__`.
Dynamic imports are outside this check.
So are attribute reads such as `module.name` that run a module-level `__getattr__`.
Relative imports are rejected, matching the Ruff configuration.
"""

import ast
import sys
from collections.abc import Iterable, Iterator, Mapping
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import NamedTuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = "langchaint"
_COMMON = "langchaint.common"


class LoadTimeExclusion(NamedTuple):
    """Modules that importing `entry` must not load, directly or through other langchaint modules.

    An excluded name also excludes its submodules.
    """

    entry: str
    excluded: tuple[str, ...]


LOAD_TIME_EXCLUSIONS = (
    # Applications that only read results import `langchaint` without any provider SDK.
    LoadTimeExclusion(
        entry="langchaint",
        excluded=(
            "langchaint.anthropic",
            "langchaint.cohere",
            "langchaint.deepseek",
            "langchaint.gemini",
            "langchaint.openai",
        ),
    ),
    # OpenAI generation works without numpy, which only OpenAI embeddings need.
    LoadTimeExclusion(entry="langchaint.openai", excluded=("numpy",)),
    # Span parsing works without OpenTelemetry installed.
    LoadTimeExclusion(entry="langchaint.span_parsing", excluded=("opentelemetry",)),
)


def _is_internal(module: str) -> bool:
    return module == _PACKAGE or module.startswith(f"{_PACKAGE}.")


def _imported_modules(nodes: Iterable[ast.AST], modules: Mapping[str, Path]) -> set[str]:
    """Return every module the import statements among `nodes` name, internal or external.

    `from package import name` also names `package.name` when that is a langchaint module.

    Raises:
        ValueError: An import is relative.
    """
    imported: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                raise ValueError("Relative imports are forbidden")
            imported.add(node.module)
            submodules = (f"{node.module}.{alias.name}" for alias in node.names)
            imported.update(submodule for submodule in submodules if submodule in modules)
    return imported


def _load_time_import_nodes(nodes: Iterable[ast.AST]) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield the import statements among `nodes` that run when the module loads.

    Function bodies run only when called, and `if TYPE_CHECKING:` bodies never run.

    Yields:
        Each import statement outside function bodies and `if TYPE_CHECKING:` bodies.
    """
    for node in nodes:
        match node:
            case ast.Import() | ast.ImportFrom():
                yield node
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                pass
            case ast.If(test=ast.Name(id="TYPE_CHECKING") | ast.Attribute(attr="TYPE_CHECKING")):
                yield from _load_time_import_nodes(node.orelse)
            case _:
                yield from _load_time_import_nodes(ast.iter_child_nodes(node))


def _getattr_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Yield every node of the module-level `__getattr__`, if the module defines one.

    Yields:
        Each node inside the `__getattr__` definition.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            yield from ast.walk(node)


class _LoadTimeImports(NamedTuple):
    """The imports of one module that the load-time check follows."""

    at_load: set[str]
    """Modules that the import statements run at load name."""
    from_at_load: set[str]
    """Modules that the `from module import name` statements run at load name."""
    in_getattr: set[str]
    """Modules that the module-level `__getattr__` imports."""


def _load_time_imports(tree: ast.Module, modules: Mapping[str, Path]) -> _LoadTimeImports:
    """Collect the imports of `tree` that the load-time check follows.

    Raises:
        ValueError: An import is relative.
    """
    load_time_nodes = list(_load_time_import_nodes(tree.body))
    return _LoadTimeImports(
        at_load=_imported_modules(load_time_nodes, modules),
        from_at_load={
            node.module
            for node in load_time_nodes
            if isinstance(node, ast.ImportFrom) and node.module is not None
        },
        in_getattr=_imported_modules(_getattr_nodes(tree), modules),
    )


def _loaded_by(entry: str, load_time_targets: Mapping[str, set[str]]) -> set[str]:
    """Return every module that importing `entry` loads, stopping at external modules.

    Loading a langchaint module first loads its parent packages.
    """
    loaded: set[str] = set()
    pending = [entry]
    while pending:
        module = pending.pop()
        if module in loaded:
            continue
        loaded.add(module)
        if module in load_time_targets:
            pending.extend(load_time_targets[module])
            if "." in module:
                pending.append(module.rpartition(".")[0])
    return loaded


def _is_excluded(module: str, excluded: tuple[str, ...]) -> bool:
    return any(module == name or module.startswith(f"{name}.") for name in excluded)


def _load_time_violation(
    exclusions: Iterable[LoadTimeExclusion], load_time_imports: Mapping[str, _LoadTimeImports]
) -> str | None:
    """Return the diagnostic for the first violated exclusion, or `None` when every exclusion holds."""
    # `from source import name` runs `source.__getattr__` for a name that `source` does not define.
    # The check cannot tell which names those are, so it counts every import in that `__getattr__`.
    load_time_targets = {
        module: imports.at_load.union(
            *(
                load_time_imports[source].in_getattr
                for source in imports.from_at_load & load_time_imports.keys()
            )
        )
        for module, imports in load_time_imports.items()
    }
    for exclusion in exclusions:
        if exclusion.entry not in load_time_targets:
            return f"Unknown load-time entry: {exclusion.entry}"
        loaded = _loaded_by(exclusion.entry, load_time_targets)
        excluded_loaded = sorted(name for name in loaded if _is_excluded(name, exclusion.excluded))
        if excluded_loaded:
            return f"Excluded load-time import: importing {exclusion.entry} loads {', '.join(excluded_loaded)}"
    return None


def _sibling_group(module: str) -> str | None:
    if module == _PACKAGE:
        return None
    return module.split(".")[1]


def check_architecture(
    *,
    project_root: Path = _PROJECT_ROOT,
    load_time_exclusions: Iterable[LoadTimeExclusion] = LOAD_TIME_EXCLUSIONS,
) -> int:
    """Return failure for invalid imports, forbidden dependencies, cycles, or excluded modules loaded at import.

    Args:
        project_root: Repository containing src/langchaint.
        load_time_exclusions: Modules that importing each entry module must not load.
    """
    source_root = project_root / "src"
    modules: dict[str, Path] = {}
    for path in sorted((source_root / _PACKAGE).rglob("*.py")):
        parts = path.relative_to(source_root).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if module in modules:
            print(f"Ambiguous module: {module}", file=sys.stderr)
            return 1
        modules[module] = path
    if _PACKAGE not in modules:
        print(f"Missing package: {source_root / _PACKAGE / '__init__.py'}", file=sys.stderr)
        return 1

    dependencies: dict[str, set[str]] = {}
    load_time_imports: dict[str, _LoadTimeImports] = {}
    for module, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            targets = {
                name for name in _imported_modules(ast.walk(tree), modules) if _is_internal(name)
            }
            load_time_imports[module] = _load_time_imports(tree, modules)
        except (SyntaxError, ValueError) as error:
            print(f"Invalid imports in {path}: {error}", file=sys.stderr)
            return 1
        for target in sorted(targets):
            if target not in modules:
                print(f"Unresolved internal import: {module} -> {target}", file=sys.stderr)
                return 1
            if (module == _COMMON or module.startswith(f"{_COMMON}.")) and not (
                target == _COMMON or target.startswith(f"{_COMMON}.")
            ):
                print(f"Forbidden common dependency: {module} -> {target}", file=sys.stderr)
                return 1
        dependencies[module] = targets

    siblings: dict[str, set[str]] = {}
    for source, targets in dependencies.items():
        source_group = _sibling_group(source)
        if source_group is None:
            continue
        sibling_targets = siblings.setdefault(source_group, set())
        for target in targets:
            target_group = _sibling_group(target)
            if target_group is not None and target_group != source_group:
                sibling_targets.add(target_group)
    for name, graph in (("file", dependencies), ("package", siblings)):
        try:
            TopologicalSorter(graph).prepare()
        except CycleError as error:
            print(f"Circular {name} dependency: {error}", file=sys.stderr)
            return 1

    violation = _load_time_violation(load_time_exclusions, load_time_imports)
    if violation is not None:
        print(violation, file=sys.stderr)
        return 1
    print("Dependency boundaries, file cycles, package cycles, and load-time exclusions checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_architecture())
