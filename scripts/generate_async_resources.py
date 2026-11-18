#!/usr/bin/env python3
"""Generate ``odon.async_resources`` from ``odon.resources``.

The synchronous resource wrappers are the single source of truth. Every method that performs
control IO becomes ``async`` and its IO calls are awaited; everything else (helpers, properties,
validation) is copied unchanged. Run with ``--check`` to verify the generated module is current.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python" / "src" / "odon" / "resources.py"
TARGET = ROOT / "python" / "src" / "odon" / "async_resources.py"

HEADER = (
    '"""Asynchronous, resource-oriented wrappers over Odon control methods.\n\n'
    "Generated from ``resources.py`` by ``scripts/generate_async_resources.py``; edit the\n"
    'synchronous module and regenerate instead of changing this file.\n"""\n'
)

# Names imported only for type checking, and their asynchronous counterparts.
RENAMED_IMPORTS = {
    "from .client import Client": "from .async_client import AsyncClient",
    "from .data import DataResource": "from .async_data import AsyncDataResource",
    "from .layers import LayerSpec": "from .layers import LayerSpec",
}
RENAMED_NAMES = {"Client": "AsyncClient", "DataResource": "AsyncDataResource"}

# Receivers, other than ``self`` and ``self._client``, that hold a resource object and whose
# IO methods therefore return awaitables.
RESOURCE_RECEIVERS = {"self._viewport", "left"}


class GenerationError(RuntimeError):
    pass


def _root_chain(node: ast.AST) -> str:
    """Dotted source of an attribute chain such as ``self._client.tasks``."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _functions_with_parents(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield node, item


def _own_calls(function: ast.FunctionDef):
    """Calls in a function body, excluding nested functions and lambdas."""

    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _nested_scopes(function: ast.FunctionDef):
    for node in ast.walk(function):
        if node is not function and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            yield node


def generate(source: str) -> str:
    tree = ast.parse(source)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    decorated_sync = {
        (cls.name, function.name)
        for cls, function in _functions_with_parents(tree)
        if function.decorator_list
    }

    def is_io_call(call: ast.Call, cls: ast.ClassDef, asynchronous: set[tuple[str, str]]) -> bool:
        if not isinstance(call.func, ast.Attribute):
            return False
        method = call.func.attr
        receiver = call.func.value
        chain = _root_chain(receiver)
        if chain == "self._client" or chain.startswith("self._client."):
            return True
        any_async_method = any(name == method for _, name in asynchronous)
        if chain == "self":
            return (cls.name, method) in asynchronous
        if chain in RESOURCE_RECEIVERS:
            return any_async_method
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id in class_names
        ):
            return (receiver.func.id, method) in asynchronous
        return False

    # A method is asynchronous when it performs IO directly or through another such method.
    asynchronous: set[tuple[str, str]] = set()
    changed = True
    while changed:
        changed = False
        for cls, function in _functions_with_parents(tree):
            key = (cls.name, function.name)
            if key in asynchronous or key in decorated_sync or function.name == "__init__":
                continue
            if any(is_io_call(call, cls, asynchronous) for call in _own_calls(function)):
                asynchronous.add(key)
                changed = True

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    data = source.encode("utf-8")

    def position(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    edits: list[tuple[int, int, bytes]] = []

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for cls, function in _functions_with_parents(tree):
        key = (cls.name, function.name)
        if key not in asynchronous:
            for call in _own_calls(function):
                if is_io_call(call, cls, asynchronous):
                    raise GenerationError(
                        f"{cls.name}.{function.name} performs IO but cannot be made asynchronous"
                    )
            continue
        for scope in _nested_scopes(function):
            for call in ast.walk(scope):
                if isinstance(call, ast.Call) and is_io_call(call, cls, asynchronous):
                    raise GenerationError(
                        f"{cls.name}.{function.name} performs IO inside a nested function or "
                        "lambda, which cannot await"
                    )
        start = position(function.lineno, function.col_offset)
        edits.append((start, start, b"async "))
        for call in _own_calls(function):
            if not is_io_call(call, cls, asynchronous):
                continue
            begin = position(call.lineno, call.col_offset)
            end = position(call.end_lineno, call.end_col_offset)
            parent = parents.get(call)
            needs_parentheses = (
                isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is call
            ) or (isinstance(parent, ast.Call) and parent.func is call)
            if needs_parentheses:
                edits.append((begin, begin, b"(await "))
                edits.append((end, end, b")"))
            else:
                edits.append((begin, begin, b"await "))

    # Rename the module's own classes wherever they are defined, constructed or annotated.
    renamed = dict(RENAMED_NAMES)
    renamed.update({name: f"Async{name}" for name in class_names})
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            line = lines[node.lineno - 1]
            column = line.index(f"class {node.name}") + len("class ")
            begin = position(node.lineno, len(line[:column].encode("utf-8")))
            edits.append((begin, begin + len(node.name), renamed[node.name].encode()))
        elif isinstance(node, ast.Name) and node.id in renamed:
            begin = position(node.lineno, node.col_offset)
            edits.append((begin, begin + len(node.id), renamed[node.id].encode()))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in renamed
            and node.end_lineno == node.lineno
        ):
            # A quoted forward reference such as ``"Client"``.
            begin = position(node.lineno, node.col_offset) + 1
            edits.append((begin, begin + len(node.value), renamed[node.value].encode()))

    for begin, end, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        data = data[:begin] + replacement + data[end:]
    generated = data.decode("utf-8")

    for old, new in RENAMED_IMPORTS.items():
        if old not in source:
            raise GenerationError(f"expected import is missing from resources.py: {old}")
        generated = generated.replace(old, new)
    module_docstring = ast.get_docstring(tree, clean=False)
    if module_docstring is None:
        raise GenerationError("resources.py has no module docstring")
    first = tree.body[0]
    docstring_source = "".join(lines[first.lineno - 1 : first.end_lineno])
    generated = generated.replace(docstring_source, HEADER, 1)
    ast.parse(generated)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if async_resources.py is not current"
    )
    arguments = parser.parse_args()
    generated = generate(SOURCE.read_text(encoding="utf-8"))
    if arguments.check:
        if TARGET.read_text(encoding="utf-8") != generated:
            print(
                "python/src/odon/async_resources.py is stale; run "
                "scripts/generate_async_resources.py",
                file=sys.stderr,
            )
            return 1
        return 0
    TARGET.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
