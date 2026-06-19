"""Build the prelaunch audit for the realtime demo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_prelaunch_audit import (
    RealtimeDemoPrelaunchAuditConfig,
    build_realtime_demo_prelaunch_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo prelaunch audit generation."""

    parser = argparse.ArgumentParser(description="Build the realtime demo prelaunch audit.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoPrelaunchAuditConfig.output_root)
    parser.add_argument(
        "--live-status-snapshot",
        type=Path,
        default=RealtimeDemoPrelaunchAuditConfig.live_status_snapshot,
    )
    parser.add_argument(
        "--operator-handoff-card",
        type=Path,
        default=RealtimeDemoPrelaunchAuditConfig.operator_handoff_card,
    )
    parser.add_argument("--launch-summary", type=Path, default=RealtimeDemoPrelaunchAuditConfig.launch_summary)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return 50 when the prelaunch audit is not ready for an operator live attempt.",
    )
    args = parser.parse_args(argv)

    audit = build_realtime_demo_prelaunch_audit(
        RealtimeDemoPrelaunchAuditConfig(
            output_root=args.output_root,
            live_status_snapshot=args.live_status_snapshot,
            operator_handoff_card=args.operator_handoff_card,
            launch_summary=args.launch_summary,
        )
    )
    print(json.dumps(audit, indent=2))
    if args.strict_exit_code and audit.get("ready_for_operator_live_attempt") is not True:
        return 50
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
