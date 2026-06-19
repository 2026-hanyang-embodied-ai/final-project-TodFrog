"""Build the guarded live-retake readiness audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_guarded_retake_readiness import (
    RealtimeDemoGuardedRetakeReadinessConfig,
    build_realtime_demo_guarded_retake_readiness,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for guarded live-retake readiness auditing."""

    parser = argparse.ArgumentParser(description="Build the guarded realtime demo retake readiness audit.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoGuardedRetakeReadinessConfig.output_root)
    parser.add_argument(
        "--readiness-summary",
        type=Path,
        default=RealtimeDemoGuardedRetakeReadinessConfig.readiness_summary,
    )
    parser.add_argument(
        "--live-status-snapshot",
        type=Path,
        default=RealtimeDemoGuardedRetakeReadinessConfig.live_status_snapshot,
    )
    parser.add_argument("--prelaunch-audit", type=Path, default=RealtimeDemoGuardedRetakeReadinessConfig.prelaunch_audit)
    parser.add_argument(
        "--operator-command-audit",
        type=Path,
        default=RealtimeDemoGuardedRetakeReadinessConfig.operator_command_audit,
    )
    parser.add_argument(
        "--live-artifact-cleanup",
        type=Path,
        default=RealtimeDemoGuardedRetakeReadinessConfig.live_artifact_cleanup,
    )
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoGuardedRetakeReadinessConfig.launch_summary)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return 55 when the guarded retake readiness audit is blocked.",
    )
    args = parser.parse_args(argv)

    summary = build_realtime_demo_guarded_retake_readiness(
        RealtimeDemoGuardedRetakeReadinessConfig(
            output_root=args.output_root,
            readiness_summary=args.readiness_summary,
            live_status_snapshot=args.live_status_snapshot,
            prelaunch_audit=args.prelaunch_audit,
            operator_command_audit=args.operator_command_audit,
            live_artifact_cleanup=args.live_artifact_cleanup,
            launch_summary=args.launch_summary,
        )
    )
    print(json.dumps(summary, indent=2))
    if args.strict_exit_code and summary.get("ready_for_guarded_retake") is not True:
        return 55
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
