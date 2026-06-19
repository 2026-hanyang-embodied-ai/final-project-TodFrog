"""Build the post-run realtime demo learning queue."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_learning_queue import (
    RealtimeDemoLearningQueueConfig,
    build_realtime_demo_learning_queue,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo learning queue generation."""

    parser = argparse.ArgumentParser(description="Build the realtime demo learning queue.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoLearningQueueConfig.output_root)
    parser.add_argument("--final-run-card", type=Path, default=RealtimeDemoLearningQueueConfig.final_run_card)
    parser.add_argument("--triage-summary", type=Path, default=RealtimeDemoLearningQueueConfig.triage_summary)
    parser.add_argument("--operator-outcome", type=Path, default=RealtimeDemoLearningQueueConfig.operator_outcome)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_learning_queue(
        RealtimeDemoLearningQueueConfig(
            output_root=args.output_root,
            final_run_card=args.final_run_card,
            triage_summary=args.triage_summary,
            operator_outcome=args.operator_outcome,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
