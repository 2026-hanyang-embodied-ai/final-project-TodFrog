"""Write local PowerShell launch scripts for the current-best realtime demo."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_launch_scripts import (
    RealtimeDemoLaunchScriptsConfig,
    write_realtime_demo_launch_scripts,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for writing realtime demo launch scripts."""

    parser = argparse.ArgumentParser(description="Write realtime demo PowerShell launch scripts.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/realtime_demo_launch_20260616"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--config", type=Path, default=Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml"))
    parser.add_argument("--sample-video", type=Path, default=Path("artifacts/realtime_demo_rehearsal_20260616/sample_input.mp4"))
    parser.add_argument("--rehearsal-output", type=Path, default=Path("artifacts/realtime_demo_rehearsal_20260616/video_rehearsal_overlay.mp4"))
    parser.add_argument("--live-output", type=Path, default=Path("artifacts/realtime_demo_rehearsal_20260616/live_camera_overlay.mp4"))
    parser.add_argument(
        "--response-preview-image",
        type=Path,
        default=Path(
            "artifacts/schunk_response_preview_v4_late_geometry_new15_20260616/"
            "frames/frame_0002_test_scissors_WIN_20260611_00000016_00000027_00000032_Pro_-_Trim.png"
        ),
    )
    parser.add_argument("--live-composite-output-root", type=Path, default=Path("artifacts/realtime_schunk_live_demo_composite_20260616"))
    parser.add_argument("--scissors-collection-output-root", type=Path, default=Path("artifacts/realtime_scissors_pose_collection_20260617"))
    parser.add_argument("--scissors-collection-config", type=Path, default=Path("configs/realtime_two_stage_selector_scissors_collection.yaml"))
    parser.add_argument("--scissors-collection-max-frames", type=int, default=3600)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-max-frames", type=int, default=900)
    args = parser.parse_args(argv)

    summary = write_realtime_demo_launch_scripts(
        RealtimeDemoLaunchScriptsConfig(
            output_root=args.output_root,
            project_root=args.project_root,
            python_executable=args.python_executable,
            config_path=args.config,
            sample_video=args.sample_video,
            rehearsal_output=args.rehearsal_output,
            live_output=args.live_output,
            response_preview_image=args.response_preview_image,
            live_composite_output_root=args.live_composite_output_root,
            scissors_collection_output_root=args.scissors_collection_output_root,
            scissors_collection_config_path=args.scissors_collection_config,
            scissors_collection_max_frames=int(args.scissors_collection_max_frames),
            camera_index=int(args.camera),
            camera_max_frames=int(args.camera_max_frames),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
