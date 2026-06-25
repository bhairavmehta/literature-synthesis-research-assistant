"""Gate 3 -- source allowlist + read-only guard.

Only allowlisted scholarly sources may enter the evidence set, and any attempt
to use a write-capable tool raises immediately.
"""
from __future__ import annotations
from ..config import SETTINGS


class ReadOnlyViolation(RuntimeError):
    pass


def assert_allowed(source: str) -> bool:
    if not any(source.endswith(dom) or source == dom for dom in SETTINGS.source_allowlist):
        raise ReadOnlyViolation(f"source not in allowlist: {source}")
    return True


def assert_read_only(tool_name: str) -> None:
    for bad in ("write", "delete", "update", "insert", "post", "put", "exec_shell"):
        if bad in tool_name.lower():
            raise ReadOnlyViolation(f"write-capable tool blocked: {tool_name}")
