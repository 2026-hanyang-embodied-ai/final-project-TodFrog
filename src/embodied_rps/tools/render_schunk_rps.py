"""Render SCHUNK SVH RPS preview images and multi-view metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.schunk import load_multiview_capture_config, load_schunk_pose_config, render_schunk_pose_previews


def main(argv: Sequence[str] | None = None) -> int:
    """Render configured SCHUNK RPS preview images."""

    parser = argparse.ArgumentParser(description="Render SCHUNK SVH RPS multi-view previews.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/isaac_multiview_capture.yaml")
    args = parser.parse_args(argv)
    config = load_multiview_capture_config(args.config)
    pose_config = load_schunk_pose_config(config.pose_config_path)
    output = render_schunk_pose_previews(
        pose_config=pose_config,
        out_dir=config.out_dir,
        metadata_path=config.metadata_path,
        yaw_degrees=config.yaw_degrees,
        pitch_degrees=config.pitch_degrees,
        gestures=config.gestures,
        distance_m=config.distance_m,
        focal_length_mm=config.focal_length_mm,
        image_width=config.image_width,
        image_height=config.image_height,
    )
    print(json.dumps({"preview_images": [path.as_posix() for path in output.preview_images], "metadata_path": output.metadata_path.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
