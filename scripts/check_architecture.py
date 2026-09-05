"""Check explicit source imports without executing langchaint.

Include conditional and function-local imports.
Dynamic imports are outside this check.
Relative imports are rejected, matching the Ruff configuration.
"""

import ast
import sys
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = "langchaint"
_COMMON = "langchaint.common"


def _is_internal(module: str) -> bool:
    return module == _PACKAGE or module.startswith(f"{_PACKAGE}.")


def _import_targets(tree: ast.Module, modules: dict[str, Path]) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names if _is_internal(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Relative imports are forbidden")
            if node.module is None or not _is_internal(node.module):
                continue
            targets.add(node.module)
            for alias in node.names:
                submodule = f"{node.module}.{alias.name}"
                if submodule in modules:
                    targets.add(submodule)
    return targets


def _sibling_group(module: str) -> str | None:
    if module == _PACKAGE:
        return None
    return module.split(".")[1]


def check_architecture(*, project_root: Path = _PROJECT_ROOT) -> int:
    """Return failure for invalid imports, forbidden dependencies, or dependency cycles.

    Args:
        project_root: Repository containing src/langchaint.
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
    for module, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            targets = _import_targets(tree, modules)
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
    print("Dependency boundaries, file cycles, and package cycles checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_architecture())
