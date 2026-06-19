"""CLI for auditing realtime demo operator-facing commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from embodied_rps.realtime_demo_operator_command_audit import (
    RealtimeDemoOperatorCommandAuditConfig,
    audit_realtime_demo_operator_commands,
)


def main(argv: list[str] | None = None) -> int:
    """Run the operator-command drift audit."""

    parser = argparse.ArgumentParser(description="Audit realtime demo operator-facing commands.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.output_root)
    parser.add_argument("--triage-summary", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.triage_summary)
    parser.add_argument(
        "--review-packet-manifest",
        type=Path,
        default=RealtimeDemoOperatorCommandAuditConfig.review_packet_manifest,
    )
    parser.add_argument("--acceptance-report", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.acceptance_report)
    parser.add_argument("--operator-outcome", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.operator_outcome)
    parser.add_argument("--live-run-checklist", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.live_run_checklist)
    parser.add_argument("--final-run-card", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.final_run_card)
    parser.add_argument("--learning-queue", type=Path, default=RealtimeDemoOperatorCommandAuditConfig.learning_queue)
    parser.add_argument(
        "--goal-progress-audit",
        type=Path,
        default=RealtimeDemoOperatorCommandAuditConfig.goal_progress_audit,
    )
    parser.add_argument(
        "--live-status-snapshot",
        type=Path,
        default=RealtimeDemoOperatorCommandAuditConfig.live_status_snapshot,
    )
    parser.add_argument(
        "--operator-handoff-card",
        type=Path,
        default=RealtimeDemoOperatorCommandAuditConfig.operator_handoff_card,
    )
    parser.add_argument("--strict-exit-code", action="store_true")
    args = parser.parse_args(argv)
    summary = audit_realtime_demo_operator_commands(
        RealtimeDemoOperatorCommandAuditConfig(
            output_root=args.output_root,
            triage_summary=args.triage_summary,
            review_packet_manifest=args.review_packet_manifest,
            acceptance_report=args.acceptance_report,
            operator_outcome=args.operator_outcome,
            live_run_checklist=args.live_run_checklist,
            final_run_card=args.final_run_card,
            learning_queue=args.learning_queue,
            goal_progress_audit=args.goal_progress_audit,
            live_status_snapshot=args.live_status_snapshot,
            operator_handoff_card=args.operator_handoff_card,
        )
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code and summary.get("audit_status") != "passed":
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
