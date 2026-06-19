"""Summarize actuator-feasible episode JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.episode import summarize_episode_log


def main(argv: Sequence[str] | None = None) -> int:
    """Write summary metrics from an episode JSONL log."""

    parser = argparse.ArgumentParser(description="Summarize RPS episode policy logs.")
    parser.add_argument("--log", required=True, type=Path, help="Path to logs/episode_policy.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Path to write summary JSON")
    args = parser.parse_args(argv)

    summary = summarize_episode_log(log_path=args.log, out_path=args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
