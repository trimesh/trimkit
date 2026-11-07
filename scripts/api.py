"""
Render the public API from the generated stubs, and what's compiled in from
trimkit.toml, into README.md, between the `<!-- api:start -->` and
`<!-- api:end -->` markers.

The stubs are generated from the compiled extension by `nanobind.stubgen`, so
the C++ bindings stay the one description of the API: this only reformats
them, as a table of every public name with its docstring's summary and the
call its numpydoc `Examples` section makes, then a table of the enums with
their members. Every callable's docstring must have `Parameters` matching
its signature, a `Returns` section and an `Examples` section, or rendering
fails.

`write` rewrites the README block (`uv run scripts generate`), and `check`
fails if it is stale (`uv run scripts stubs`).

Standard library only.
"""

import ast
import re
import sys

from builder import ROOT, load

STUBS = ROOT / "src" / "trimkit"
README = ROOT / "README.md"
START, END = "<!-- api:start -->", "<!-- api:end -->"


def returns(doc: str, name: str) -> str:
    """
    The result names from a numpydoc `Returns` section, as `name` or
    `(a, b, c)`: every documented callable must name what it returns.
    """
    match = re.search(r"Returns\n-+\n(.*?)(\n\S[^\n]*\n-+\n|\Z)", doc, re.S)
    if match is None:
        raise ValueError(f"{name}: its docstring has no Returns section")
    names = re.findall(r"^(\w+) :", match[1], re.M)
    if not names:
        raise ValueError(f"{name}: its Returns section names nothing")
    return names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"


def check_parameters(function: ast.FunctionDef, doc: str, name: str) -> None:
    """
    The numpydoc `Parameters` section names exactly the signature's arguments,
    in order: the docs and the binding can't drift apart.
    """
    arguments = function.args
    expected = [a.arg for a in [*arguments.args, *arguments.kwonlyargs] if a.arg != "self"]
    match = re.search(r"Parameters\n-+\n(.*?)(\n\S[^\n]*\n-+\n|\Z)", doc, re.S)
    documented = re.findall(r"^(\w+) :", match[1], re.M) if match else []
    if documented != expected:
        raise ValueError(f"{name}: its docstring documents {documented} but its signature takes {expected}")


def check_documented(
    function: ast.FunctionDef, name: str, constructor: bool, documentation: str | None = None
) -> None:
    # Parameters match the signature, and all but a constructor name what they return
    doc = (documentation if constructor else ast.get_docstring(function)) or ""
    check_parameters(function, doc, name)
    if not constructor:
        returns(doc, name)


def first_line(doc: str | None) -> str:
    # the summary paragraph, however it wraps
    return " ".join((doc or "").strip().split("\n\n")[0].split())


def example(doc: str, name: str) -> str:
    """
    The call an `Examples` section makes: its last `>>>` line calling `name`
    (a method or property by its own name), else its last naming it. Every
    public callable has one.
    """
    match = re.search(r"Examples\n-+\n(.*)", doc, re.S)
    lines = [line[4:] for line in (match[1] if match else "").splitlines() if line.startswith(">>> ")]
    own = re.escape(name.rsplit(".", 1)[-1])
    named = [c for c in lines if re.search(rf"\b{own}\(", c)] or [
        c for c in lines if re.search(rf"\b{own}\b", c)
    ]
    if not named:
        raise ValueError(f"{name}: its docstring has no Examples section calling it")
    return named[-1]


def attributes(doc: str) -> dict[str, str]:
    # a numpydoc Attributes section: each name and its description, one line
    match = re.search(r"Attributes\n-+\n(.*?)(\n\S[^\n]*\n-+\n|\Z)", doc, re.S)
    found, name = {}, None
    for line in (match[1] if match else "").splitlines():
        if line and not line[0].isspace():
            name = line.split(" :")[0].strip()
            found[name] = ""
        elif name:
            found[name] = (found[name] + " " + line.strip()).strip()
    return found


def row(name: str, summary: str, call: str) -> str:
    return f"| `trimkit.{name}` | {summary} | {f'`{call}`' if call else ''} |"


def rows(path, prefix: str) -> list[str]:
    # one row per callable, method and described name a stub file holds
    tree = ast.parse(path.read_text())
    module = ast.get_docstring(tree) or ""
    lines = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = ast.get_docstring(node) or ""
            check_documented(node, node.name, False)
            lines.append(row(prefix + node.name, first_line(doc), example(doc, node.name)))
        elif isinstance(node, ast.ClassDef) and not any(
            ast.unparse(b).endswith("IntEnum") for b in node.bases
        ):
            doc = ast.get_docstring(node) or ""
            init = next(
                (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
            )
            if init:
                check_documented(init, node.name, True, doc)
            lines.append(row(prefix + node.name, first_line(doc), example(doc, node.name)))
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and not method.name.startswith("_"):
                    name = f"{node.name}.{method.name}"
                    method_doc = ast.get_docstring(method) or ""
                    if not any(ast.unparse(d) == "property" for d in method.decorator_list):
                        check_documented(method, name, False)
                    lines.append(row(prefix + name, first_line(method_doc), example(method_doc, name)))
    # names that are no callable, which the module's Attributes section describes:
    # all but the submodule and the enums, which have a table of their own
    for name, text in attributes(module).items():
        if name != "exchange" and not name.endswith("Kind"):
            lines.append(row(prefix + name, text, example(module, name)))
    return lines


def render() -> str:
    lines = ["| `trimkit.X` | what it is | example |", "|---|---|---|"]
    lines += rows(STUBS / "__init__.pyi", "") + rows(STUBS / "exchange.pyi", "exchange.")
    lines += ["", "| `trimkit.exchange.X` | what it is | members |", "|---|---|---|"]
    for node in ast.parse((STUBS / "exchange.pyi").read_text()).body:
        if isinstance(node, ast.ClassDef) and any(ast.unparse(b).endswith("IntEnum") for b in node.bases):
            members = [n.targets[0].id for n in node.body if isinstance(n, ast.Assign)]
            members += [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
            names = ", ".join(f"`{m}`" for m in members)
            lines.append(
                f"| `trimkit.exchange.{node.name}` | {first_line(ast.get_docstring(node))} | {names} |"
            )
    return "\n".join(lines) + "\n"


def inside() -> str:
    # what each part compiles in, with its license, from trimkit.toml
    manifest = load()

    def links(upstreams) -> str:
        return ", ".join("ours" if u.own else f"[{u.repo}](https://{u.host}/{u.repo})" for u in upstreams)

    def spdx(upstreams) -> str:
        return ", ".join(u.spdx for u in upstreams)

    lines = ["| part | upstream | license |", "|---|---|---|"]
    lines += [f"| `{p.name}` | {links(p.upstreams)} | {spdx(p.upstreams)} |" for p in manifest.parts.values()]
    every = manifest.extension_upstreams
    lines.append(f"| every part | {links(every)} | {spdx(every)} |")
    return "\n".join(lines) + "\n"


def block() -> str:
    note = "<!-- generated by `uv run scripts generate` from the docstrings and trimkit.toml: don't edit -->"
    return f"{START}\n{note}\n\n{render()}\n{inside()}{END}"


def updated(text: str) -> str:
    if START not in text or END not in text:
        raise ValueError(f"README.md has no {START} ... {END} markers")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    return before + block() + after


def write() -> None:
    README.write_text(updated(README.read_text()))


def check() -> None:
    text = README.read_text()
    if updated(text) != text:
        sys.exit("README.md's API summary is stale: run `uv run scripts generate`")
