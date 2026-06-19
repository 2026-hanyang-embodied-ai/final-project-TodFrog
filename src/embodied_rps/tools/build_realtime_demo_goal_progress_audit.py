"""Build the objective-level realtime demo goal progress audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_goal_progress_audit import (
    RealtimeDemoGoalProgressAuditConfig,
    build_realtime_demo_goal_progress_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo goal progress auditing."""

    parser = argparse.ArgumentParser(description="Build the realtime demo goal progress audit.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoGoalProgressAuditConfig.output_root)
    parser.add_argument("--readiness-summary", type=Path, default=RealtimeDemoGoalProgressAuditConfig.readiness_summary)
    parser.add_argument("--evidence-bundle", type=Path, default=RealtimeDemoGoalProgressAuditConfig.evidence_bundle)
    parser.add_argument("--final-run-card", type=Path, default=RealtimeDemoGoalProgressAuditConfig.final_run_card)
    parser.add_argument("--learning-queue", type=Path, default=RealtimeDemoGoalProgressAuditConfig.learning_queue)
    parser.add_argument("--final-candidate", type=Path, default=RealtimeDemoGoalProgressAuditConfig.final_candidate)
    parser.add_argument("--archive-index", type=Path, default=RealtimeDemoGoalProgressAuditConfig.archive_index)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return a non-zero status when the objective is not complete.",
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_goal_progress_audit(
        RealtimeDemoGoalProgressAuditConfig(
            output_root=args.output_root,
            readiness_summary=args.readiness_summary,
            evidence_bundle=args.evidence_bundle,
            final_run_card=args.final_run_card,
            learning_queue=args.learning_queue,
            final_candidate=args.final_candidate,
            archive_index=args.archive_index,
        )
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code:
        return _strict_exit_code(summary)
    return 0


def _strict_exit_code(summary: dict[str, object]) -> int:
    if summary.get("goal_complete") is True:
        return 0
    status = str(summary.get("goal_status") or "")
    if status == "incomplete_awaiting_live_capture":
        return 10
    if status == "incomplete_postprocess_repair_needed":
        return 30
    if status == "incomplete_research_iteration_needed":
        return 40
    return 60


if __name__ == "__main__":
    raise SystemExit(main())
