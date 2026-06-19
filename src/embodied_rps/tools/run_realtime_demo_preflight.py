"""Run preflight checks for the current-best realtime demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_preflight import (
    RealtimeDemoPreflightConfig,
    run_realtime_demo_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for realtime demo preflight checks."""

    parser = argparse.ArgumentParser(description="Run realtime demo preflight checks.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml"))
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_preflight_20260616"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--check-camera", action="store_true")
    parser.add_argument("--check-hand-visibility", action="store_true")
    parser.add_argument("--hand-visibility-max-frames", type=int, default=60)
    parser.add_argument("--hand-visibility-min-detection-rate", type=float, default=0.80)
    parser.add_argument("--require-response-prompt", default="scissors")
    parser.add_argument("--allow-missing-reset-on-prompt-change", action="store_true")
    args = parser.parse_args(argv)

    summary = run_realtime_demo_preflight(
        RealtimeDemoPreflightConfig(
            project_root=args.project_root,
            config_path=args.config,
            python_executable=args.python_executable,
            output_root=args.output_root,
            camera_index=int(args.camera),
            check_camera=bool(args.check_camera),
            check_hand_visibility=bool(args.check_hand_visibility),
            hand_visibility_max_frames=int(args.hand_visibility_max_frames),
            hand_visibility_min_detection_rate=float(args.hand_visibility_min_detection_rate),
            require_response_prompt=args.require_response_prompt,
            require_reset_on_prompt_change=not bool(args.allow_missing_reset_on_prompt_change),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if bool(summary["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
