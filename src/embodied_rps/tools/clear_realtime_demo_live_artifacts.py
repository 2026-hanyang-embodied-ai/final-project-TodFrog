"""CLI for clearing stale realtime demo live artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_live_artifact_cleanup import (
    RealtimeDemoLiveArtifactCleanupConfig,
    clear_realtime_demo_live_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Clear configured fixed-path live artifacts before a new camera run."""

    parser = argparse.ArgumentParser(description="Clear stale realtime demo live artifacts.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.output_root)
    parser.add_argument("--workspace-root", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.workspace_root)
    parser.add_argument("--live-overlay", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.live_overlay)
    parser.add_argument("--live-frame-log", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.live_frame_log)
    parser.add_argument("--live-postcapture-root", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.live_postcapture_root)
    parser.add_argument("--live-composite-root", type=Path, default=RealtimeDemoLiveArtifactCleanupConfig.live_composite_root)
    parser.add_argument(
        "--live-overlay-contract-root",
        type=Path,
        default=RealtimeDemoLiveArtifactCleanupConfig.live_overlay_contract_root,
    )
    parser.add_argument(
        "--live-rock-retake-gate-root",
        type=Path,
        default=RealtimeDemoLiveArtifactCleanupConfig.live_rock_retake_gate_root,
    )
    args = parser.parse_args(argv)

    summary = clear_realtime_demo_live_artifacts(
        RealtimeDemoLiveArtifactCleanupConfig(
            output_root=args.output_root,
            workspace_root=args.workspace_root,
            live_overlay=args.live_overlay,
            live_frame_log=args.live_frame_log,
            live_postcapture_root=args.live_postcapture_root,
            live_composite_root=args.live_composite_root,
            live_overlay_contract_root=args.live_overlay_contract_root,
            live_rock_retake_gate_root=args.live_rock_retake_gate_root,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
