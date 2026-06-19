"""Run the current-best realtime RPS skeleton demo configuration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_launcher import build_realtime_demo_argv, load_realtime_demo_config
from embodied_rps.tools.run_realtime_skeleton_predictor import main as run_realtime_skeleton_predictor


def main(argv: Sequence[str] | None = None) -> int:
    """Run or dry-run the current-best realtime demo launcher."""

    parser = argparse.ArgumentParser(description="Run the current-best realtime RPS skeleton demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue.yaml"),
        help="Realtime demo launcher YAML config.",
    )
    parser.add_argument("--video", type=Path, default=None, help="Prerecorded MP4 input for dry-run/demo validation.")
    parser.add_argument("--camera", type=int, default=None, help="Live camera index.")
    parser.add_argument("--output", type=Path, default=None, help="Optional overlay MP4 output.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional root for dry-run launcher artifacts.")
    parser.add_argument("--frame-log-jsonl", type=Path, default=None, help="Optional per-frame inference log JSONL output.")
    parser.add_argument("--skeleton-npz", type=Path, default=None, help="Optional per-frame canonical skeleton sidecar NPZ output.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit.")
    parser.add_argument(
        "--expected-actual-gesture",
        choices=("rock", "paper", "scissors"),
        default=None,
        help="Operator ground-truth gesture used only for frame-log and post-capture validation metadata.",
    )
    parser.add_argument(
        "--collection-mode",
        action="store_true",
        help="Disable final-demo auto-stop behavior so a long pose-collection session can be recorded.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated realtime predictor argv and exit.")
    args = parser.parse_args(argv)

    config = load_realtime_demo_config(args.config)
    delegated_argv = build_realtime_demo_argv(
        config,
        video=args.video,
        camera=args.camera,
        output=args.output,
        frame_log_jsonl=args.frame_log_jsonl,
        skeleton_npz=args.skeleton_npz,
        max_frames=args.max_frames,
        expected_actual_gesture=args.expected_actual_gesture,
        collection_mode=bool(args.collection_mode),
    )
    if args.dry_run:
        payload = {
            "config": str(args.config),
            "output_root": str(args.output_root) if args.output_root is not None else None,
            "argv": delegated_argv,
        }
        if args.output_root is not None:
            args.output_root.mkdir(parents=True, exist_ok=True)
            (args.output_root / "current_best_realtime_demo_dry_run.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(payload, indent=2))
        return 0
    return run_realtime_skeleton_predictor(delegated_argv)


if __name__ == "__main__":
    raise SystemExit(main())
