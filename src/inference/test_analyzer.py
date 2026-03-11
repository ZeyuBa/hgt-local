"""Test analysis report helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestAnalyzer:
    """Minimal analysis surface for test predictions."""

    summary: dict

    def to_markdown(self) -> str:
        lines = ["# Test Analysis", ""]
        for key, value in sorted(self.summary.items()):
            lines.append(f"- `{key}`: {value}")
        return "\n".join(lines) + "\n"

