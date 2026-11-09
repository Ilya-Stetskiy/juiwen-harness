"""Trivial deterministic solver: a task that is one bare arithmetic expression ("What is 12 * (3 + 4)?").

Returns None whenever it is not sure: no recognised sentence, unsafe characters, a non-integer result, other check type.
"""
from __future__ import annotations

import ast
import operator
import re

_SENT = re.compile(r"(?:what is|compute|calculate|evaluate|find the value of)\s*:?\s*([-+*/().\d\s^]+?)\s*[?.=]*\s*$", re.I)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 12 or abs(a) > 10 ** 6):
            raise ValueError("too large")
        return _OPS[type(node.op)](a, b)
    raise ValueError("unsupported")


def solve(task) -> str | None:
    if task.check != "exact_int":
        return None
    for sent in re.split(r"(?<=[.!?])\s+|\n", task.instruction.strip()):
        m = _SENT.search(sent.strip())
        if not m or not re.search(r"\d", m.group(1)):
            continue
        try:
            v = _eval(ast.parse(m.group(1).replace("^", "**").strip(), mode="eval"))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None
        if isinstance(v, float):
            if not v.is_integer():
                return None
            v = int(v)
        return str(v)
    return None
