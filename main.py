"""Unified training and inference entrypoint."""

from __future__ import annotations

import argparse
import sys

from src.training.config import EXECUTION_MODE_CHOICES, RuntimeConfigError, SMOKE_SUCCESS_PROMISE
from src.training.trainer import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HGT alarm pipeline")
    parser.add_argument("--config", required=True, help="Path to the runtime YAML config")
    parser.add_argument(
        "--mode",
        default="train",
        choices=EXECUTION_MODE_CHOICES,
        help="Run full training or checkpoint-backed inference",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for inference mode; defaults to the configured best checkpoint path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_pipeline(
            args.config,
            "full",
            mode=args.mode,
            checkpoint_path=args.checkpoint_path,
        )
    except (OSError, RuntimeConfigError, ValueError, FileNotFoundError) as exc:
        print(f"error={exc}", file=sys.stderr, flush=True)
        return 1

    print(SMOKE_SUCCESS_PROMISE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
