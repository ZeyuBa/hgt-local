#!/usr/bin/env python3
"""Extract final validation metrics from run.log emitted by HF Trainer."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run.log")
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}")
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"\{[^{}]*eval_edge_best_f1[^{}]*\}", text)
    if not blocks:
        print("ERROR: no eval metrics found in run.log")
        return 2

    try:
        last = ast.literal_eval(blocks[-1])
    except (ValueError, SyntaxError) as exc:
        print(f"ERROR: failed to parse final eval block: {exc}")
        return 3

    print(f"val_f1_cal\t{last.get('eval_edge_best_f1', 'N/A')}")
    print(f"val_graph_acc\t{last.get('eval_graph_accuracy', 'N/A')}")
    print(f"val_auc\t{last.get('eval_edge_auc', 'N/A')}")
    print(f"val_loss\t{last.get('eval_loss', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
