"""Emit ALL_NAMES.md: every identifier defined under src/langchaint, by AST.

The output is a naming-review inventory: classes, functions, methods, properties,
type aliases, module constants, class fields, instance attributes, and parameters,
each flagged public or private (leading underscore, dunders excepted).
Parameters and fields are included because most of the package's naming rules govern
them (system_prompt not system, exact_name=exact_name, input_tokens_* family, unit
suffixes like _seconds), so a review that omitted them would miss its main target.

Regenerate after any rename: `uv run python -m scripts.dump_all_names` from the repo
root, or `uv run python scripts/dump_all_names.py`. Zero arguments; the source root
and output path are the constants below, so the run is committed, not assembled at the
invocation.
"""

import ast
import pathlib
from collections.abc import Callable, Sequence

SOURCE_ROOT = pathlib.Path("src/langchaint")
OUTPUT_PATH = pathlib.Path("ALL_NAMES.md")


def is_private(name: str) -> bool:
    """Leading-underscore, but not a dunder, is private."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def flag(name: str) -> str:
    """Return the public/private marker for a class member (no export tier applies)."""
    return "priv" if is_private(name) else "pub "


def read_all_exports(tree: ast.Module) -> set[str]:
    """Return the string entries of a module's __all__, empty when it has none."""
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets)
            and isinstance(stmt.value, (ast.List, ast.Tuple))
        ):
            return {
                element.value
                for element in stmt.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    return set()


def module_name(path: pathlib.Path) -> str:
    """Dotted import path of a file under SOURCE_ROOT."""
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return "langchaint" + "".join(f".{p}" for p in parts)


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render the parameter list and return annotation, self/cls dropped."""
    args = ast.unparse(node.args)
    for lead in ("self, ", "cls, ", "self", "cls"):
        if args.startswith(lead):
            args = args[len(lead) :]
            break
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({args}){returns}"


def decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Bare decorator names, so property/override/overload can be recognized."""
    out: set[str] = set()
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            out.add(dec.id)
        elif isinstance(dec, ast.Attribute):
            out.add(dec.attr)
    return out


def instance_attrs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """self.<attr> assignment targets in a method body, first occurrence order."""
    seen: dict[str, None] = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    seen.setdefault(target.attr, None)
    return list(seen)


def render_class(node: ast.ClassDef, lines: list[str], tier: str) -> None:
    """Append one class block: fields, class vars, properties, methods.

    tier is the already-formatted export marker for the class name itself
    (members carry no tier, only pub/priv).
    """
    attrs: dict[str, None] = {}
    base = ", ".join(ast.unparse(b) for b in node.bases)
    header = f"### class `{node.name}`" + (f"({base})" if base else "")
    lines.append(f"{header}  `[{tier}]`")
    fields: list[str] = []
    classvars: list[str] = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.append(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            classvars.extend(t.id for t in stmt.targets if isinstance(t, ast.Name))
    if fields:
        lines.append("  - fields: " + ", ".join(f"`{n}` [{flag(n)}]" for n in fields))
    if classvars:
        lines.append("  - class vars: " + ", ".join(f"`{n}` [{flag(n)}]" for n in classvars))
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = decorator_names(stmt)
            kind = "property" if "property" in decs else "method"
            prefix = "async " if isinstance(stmt, ast.AsyncFunctionDef) else ""
            lines.append(
                f"  - {kind} `{prefix}{stmt.name}{signature(stmt)}` [{flag(stmt.name)}]"
            )
            for attr in instance_attrs(stmt):
                attrs.setdefault(attr, None)
    if attrs:
        lines.append("  - instance attrs: " + ", ".join(f"`{n}` [{flag(n)}]" for n in attrs))
    lines.append("")


type Tally = Callable[[str, str], None]


def collect_module_level(tree: ast.Module, tally: Tally) -> tuple[list[str], list[str]]:
    """Return a module's top-level type-alias and constant names, tallying each.

    __all__ is skipped: it is export metadata, not a domain name.
    """
    type_aliases: list[str] = []
    constants: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.TypeAlias) and isinstance(stmt.name, ast.Name):
            type_aliases.append(stmt.name.id)
            tally("type", stmt.name.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            constants.append(stmt.target.id)
            tally("const", stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id != "__all__":
                    constants.append(target.id)
                    tally("const", target.id)
    return type_aliases, constants


def tally_class(node: ast.ClassDef, tally: Tally) -> None:
    """Tally one class and each of its methods, properties, and fields."""
    tally("class", node.name)
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "property" if "property" in decorator_names(sub) else "method"
            tally(kind, sub.name)
        elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
            tally("field", sub.target.id)


def module_block(
    path: pathlib.Path, tier_of: Callable[[str], str]
) -> tuple[list[str], dict[str, int]]:
    """Return the markdown lines for one module plus its per-kind counts.

    tier_of formats the export marker of a module-level definition name.
    """
    tree = ast.parse(path.read_text())
    lines = [f"## {module_name(path)}", f"`{path.as_posix()}`", ""]
    counts: dict[str, int] = {}

    def tally(kind: str, name: str) -> None:
        counts[f"{flag(name).strip()} {kind}"] = counts.get(f"{flag(name).strip()} {kind}", 0) + 1

    type_aliases, constants = collect_module_level(tree, tally)
    if type_aliases:
        lines.append(
            "**Type aliases:** " + ", ".join(f"`{n}` [{tier_of(n)}]" for n in type_aliases)
        )
        lines.append("")
    if constants:
        lines.append("**Constants:** " + ", ".join(f"`{n}` [{tier_of(n)}]" for n in constants))
        lines.append("")

    functions = [s for s in tree.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if functions:
        lines.append("**Functions:**")
        for fn in functions:
            prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
            lines.append(f"- `{prefix}{fn.name}{signature(fn)}` [{tier_of(fn.name)}]")
            tally("function", fn.name)
        lines.append("")

    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            tally_class(stmt, tally)
            render_class(stmt, lines, tier_of(stmt.name))
    return lines, counts


def build_tier_of(files: Sequence[pathlib.Path]) -> Callable[[str], str]:
    """Return a function formatting a module-level name's export marker.

    A private name is "priv". A public name is "pub, top-level" when the top-level
    langchaint package re-exports it, "pub, subpackage" when a backend or tracing
    subpackage's __all__ carries it (including names it re-exports from a submodule),
    and "pub, module-only" when no __all__ lists it. Membership is by name, so a name
    a subpackage re-exports from a submodule is marked subpackage at its definition too.
    """
    top_level: set[str] = set()
    subpackage: set[str] = set()
    for path in files:
        exports = read_all_exports(ast.parse(path.read_text()))
        if module_name(path) == "langchaint":
            top_level |= exports
        else:
            subpackage |= exports

    def tier_of(name: str) -> str:
        if is_private(name):
            return "priv"
        if name in top_level:
            return "pub, top-level"
        if name in subpackage:
            return "pub, subpackage"
        return "pub, module-only"

    return tier_of


def main() -> None:
    """Write OUTPUT_PATH from the current source tree."""
    files = sorted(SOURCE_ROOT.rglob("*.py"))
    tier_of = build_tier_of(files)
    totals: dict[str, int] = {}
    body: list[str] = []
    for path in files:
        lines, counts = module_block(path, tier_of)
        body.extend(lines)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value

    priv_total = sum(v for k, v in totals.items() if k.startswith("priv"))
    pub_total = sum(v for k, v in totals.items() if k.startswith("pub"))
    header = [
        "# ALL_NAMES",
        "",
        "Every identifier defined under `src/langchaint`, generated by `scripts/dump_all_names.py`.",
        "A class member is flagged `[pub]` or `[priv]` (leading underscore, dunders excepted);",
        "a module-level definition also carries its export tier: `top-level` (re-exported from the",
        "`langchaint` package), `subpackage` (in a backend or tracing subpackage's `__all__`), or",
        "`module-only` (public but on no `__all__`, importable from its own module).",
        "Parameters are shown inside each signature rather than listed separately.",
        "Regenerate with `uv run python scripts/dump_all_names.py` after any rename.",
        "",
        "## Import tiers",
        "",
        "The tiers name who imports a name, a design decision the source cannot show on its own:",
        "",
        "- Applications import the neutral core from top-level `langchaint`,",
        "  and each backend's constructor, model names, pricing, and concrete adapter from its",
        "  subpackage (`langchaint.anthropic` / `langchaint.openai`).",
        "- Adapter authors import the provider contract from `langchaint.provider`",
        "  (`Provider`, `BoundProvider`, `ProviderStream`, `ProviderResult`, `Binding`, `ErrorClass`).",
        "- Top-level `__all__` re-exports only the SDK-free application surface, so the provider",
        "  constructors, pricing tables, and concrete adapters stay off it: re-exporting them would",
        "  force `import langchaint` and every downstream type check through both SDKs.",
        "",
        "## Counts",
        "",
        f"Definitions (excludes parameters and instance attrs): {pub_total + priv_total} "
        f"({priv_total} private, {pub_total} public).",
        "",
        "| kind | private | public |",
        "| --- | --- | --- |",
    ]
    kinds = sorted({k.split(" ", 1)[1] for k in totals})
    header.extend(
        f"| {kind} | {totals.get(f'priv {kind}', 0)} | {totals.get(f'pub {kind}', 0)} |"
        for kind in kinds
    )
    header.append("")

    OUTPUT_PATH.write_text("\n".join(header + body) + "\n")
    print(f"wrote {OUTPUT_PATH} ({pub_total + priv_total} definitions)")


if __name__ == "__main__":
    main()
