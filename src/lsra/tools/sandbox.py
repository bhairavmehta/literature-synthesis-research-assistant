"""Restricted numeric sandbox.

Lets the Reasoner verify a quantitative claim (e.g. recompute a percentage)
without exposing a general code path. No imports, no builtins, math only.
"""
from __future__ import annotations
import ast
import operator as op

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}


def safe_eval_numeric(expr: str) -> float:
    """Evaluate an arithmetic expression over numbers only."""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return float(_ev(ast.parse(expr, mode="eval").body))
