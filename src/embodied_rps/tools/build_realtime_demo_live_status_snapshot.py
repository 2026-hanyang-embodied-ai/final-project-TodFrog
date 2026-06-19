"""Build the live status snapshot for the realtime demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_live_status_snapshot import (
    RealtimeDemoLiveStatusSnapshotConfig,
    build_realtime_demo_live_status_snapshot,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo live status snapshot generation."""

    parser = argparse.ArgumentParser(description="Build the realtime demo live status snapshot.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.output_root)
    parser.add_argument("--goal-audit", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.goal_audit)
    parser.add_argument("--final-run-card", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.final_run_card)
    parser.add_argument("--learning-queue", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.learning_queue)
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.readiness_summary)
    parser.add_argument("--evidence-bundle", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.evidence_bundle)
    parser.add_argument("--live-overlay", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.live_overlay)
    parser.add_argument("--live-frame-log", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.live_frame_log)
    parser.add_argument("--live-composite", type=Path, default=RealtimeDemoLiveStatusSnapshotConfig.live_composite)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_live_status_snapshot(
        RealtimeDemoLiveStatusSnapshotConfig(
            output_root=args.output_root,
            goal_audit=args.goal_audit,
            final_run_card=args.final_run_card,
            learning_queue=args.learning_queue,
            readiness_summary=args.readiness_summary,
            evidence_bundle=args.evidence_bundle,
            live_overlay=args.live_overlay,
            live_frame_log=args.live_frame_log,
            live_composite=args.live_composite,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
