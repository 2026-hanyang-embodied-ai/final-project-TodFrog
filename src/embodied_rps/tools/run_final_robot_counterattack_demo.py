"""CLI for the final robot counterattack demo artifact builder."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.final_robot_counterattack_demo import (
    FINAL_DEMO_CONFIG,
    build_final_counterattack_artifacts,
    freeze_final_demo_policy,
    write_final_report_assets,
    write_render_audit_artifacts,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final robot counterattack demo artifacts.")
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--overlay-video", type=Path, default=None)
    parser.add_argument("--frame-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FINAL_DEMO_CONFIG)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--schunk-pose-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy-freeze-root",
        type=Path,
        default=Path("artifacts/final_submission_model_policy_freeze_20260619"),
    )
    parser.add_argument(
        "--render-audit-root",
        type=Path,
        default=Path("artifacts/final_submission_robot_hand_render_audit_20260619"),
    )
    parser.add_argument(
        "--render-artifact-root",
        type=Path,
        default=Path("artifacts/schunk_joint_target_skeleton_passed"),
    )
    parser.add_argument(
        "--report-assets-root",
        type=Path,
        default=Path("artifacts/final_submission_report_assets_20260619"),
    )
    parser.add_argument("--diagnostic-replay-summary", type=Path, default=None)
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    freeze_summary = freeze_final_demo_policy(
        config_path=args.config,
        output_root=args.policy_freeze_root,
        project_root=project_root,
    )
    render_audit = write_render_audit_artifacts(
        render_artifact_root=args.render_artifact_root,
        output_root=args.render_audit_root,
        project_root=project_root,
    )
    replay_summary = build_final_counterattack_artifacts(
        overlay_video=args.overlay_video,
        frame_log=args.frame_log,
        config_path=args.config,
        pose_config_path=args.pose_config,
        schunk_pose_config_path=args.schunk_pose_config,
        output_root=args.output_root,
        project_root=project_root,
        run_label=args.mode,
    )
    report_summary = write_final_report_assets(
        output_root=args.report_assets_root,
        project_root=project_root,
        replay_summary=replay_summary,
        freeze_summary=freeze_summary,
        render_audit=render_audit,
        diagnostic_replay_summary_path=args.diagnostic_replay_summary,
    )
    payload = {
        "status": "passed" if replay_summary.get("status") == "passed" else replay_summary.get("status"),
        "mode": args.mode,
        "policy_freeze": freeze_summary.get("outputs"),
        "render_audit": render_audit.get("outputs"),
        "demo": replay_summary,
        "report_assets": report_summary.get("outputs"),
        "stopped_before_final_packaging": True,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
