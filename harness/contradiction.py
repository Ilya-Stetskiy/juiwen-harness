"""Static detector of contradictory tests (zero tokens).

An unsolvable coding task can have a test file in which the same call is expected to return two different values,
for example `generate_integers(2, 10) == [2, 4, 6, 8]` in a helper check and `... == [2, 4, 6, 8, 0]` in another
test. No implementation can pass both, so the fallback agent must not be sent to such a task.

`find_contradictions(test_source)` returns a list of human-readable descriptions, empty when none is found. It only
reports a contradiction when both expected values are literals that differ, so it errs on the side of silence:
a missed contradiction costs tokens, a false one would cost a solvable task.
"""
from __future__ import annotations

import ast
from collections import defaultdict


def _name_of(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.expr):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _same(a, b) -> bool:
    try:
        return type(a) is type(b) and a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                                  and not isinstance(a, bool) and not isinstance(b, bool) and a == b)
    except Exception:  # noqa: BLE001
        return a is b


def _helper_params(tree: ast.Module) -> dict[str, dict[str, str]]:
    """helper function name -> {parameter name: real function name}, from calls like check_f(m.f)."""
    helpers = {n.name: [a.arg for a in n.args.args] for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    mapping: dict[str, dict[str, str]] = defaultdict(dict)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in helpers:
            for param, arg in zip(helpers[n.func.id], n.args):
                real = _name_of(arg)
                if real:
                    mapping[n.func.id][param] = real
    return mapping


def _equalities(body_owner: ast.AST):
    """Yield (call node, expected node) for `assert call == x`, `assert x == call` and assertEqual(call, x)."""
    for n in ast.walk(body_owner):
        pair = None
        if isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1 \
                and isinstance(n.test.ops[0], ast.Eq):
            pair = (n.test.left, n.test.comparators[0])
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "assertEqual" \
                and len(n.args) >= 2:
            pair = (n.args[0], n.args[1])
        if pair is None:
            continue
        a, b = pair
        if isinstance(a, ast.Call):
            yield a, b
        elif isinstance(b, ast.Call):
            yield b, a


def find_contradictions(test_source: str) -> list[str]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    params = _helper_params(tree)
    seen: dict[tuple[str, str], list[tuple[object, str]]] = defaultdict(list)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    scopes = [(f, params.get(f.name, {})) for f in funcs] + [(c, {}) for c in classes]
    for owner, pmap in scopes:
        for call, expected in _equalities(owner):
            name = _name_of(call.func)
            if name is None:
                continue
            name = pmap.get(name, name)
            ok, value = _literal(expected)
            if not ok:
                continue
            args = ast.dump(ast.Tuple(elts=call.args, ctx=ast.Load())) + "|" + \
                ",".join(f"{k.arg}={ast.dump(k.value)}" for k in call.keywords)
            seen[(name, args)].append((value, ast.unparse(call)))
    found = []
    for (_name, _args), items in seen.items():
        first_value, first_call = items[0]
        for value, _call in items[1:]:
            if not _same(first_value, value):
                found.append(f"{first_call} is expected to be {first_value!r} and {value!r}")
                break
    return found
