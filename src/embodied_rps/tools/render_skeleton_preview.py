"""Render SVG hand-skeleton previews from synthetic trajectory episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.visualization import render_preview_set, render_skeleton_preview


def main(argv: Sequence[str] | None = None) -> int:
    """Render one episode or one episode per class as SVG preview artifacts."""

    parser = argparse.ArgumentParser(description="Render hand-skeleton SVG previews.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to synthetic dataset .npz")
    parser.add_argument("--hand-config", required=True, type=Path, help="Path to configs/kinematic_rps.yaml")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for SVG artifacts")
    parser.add_argument("--frame-count", type=int, default=8, help="Number of frames in the preview.")
    parser.add_argument("--prefix", default="skeleton", help="Output filename prefix.")
    parser.add_argument("--episode-index", type=int, default=None, help="Render one explicit episode index.")
    parser.add_argument("--split", default="test", help="Split for per-class preview when episode index is omitted.")
    args = parser.parse_args(argv)

    if args.episode_index is None:
        artifacts = render_preview_set(
            dataset_path=args.dataset,
            hand_config_path=args.hand_config,
            out_dir=args.out_dir,
            split=str(args.split),
            frame_count=int(args.frame_count),
            prefix=str(args.prefix),
        )
    else:
        artifacts = [
            render_skeleton_preview(
                dataset_path=args.dataset,
                hand_config_path=args.hand_config,
                out_dir=args.out_dir,
                episode_index=int(args.episode_index),
                frame_count=int(args.frame_count),
                prefix=str(args.prefix),
            )
        ]
    print(
        json.dumps(
            {
                "artifacts": [
                    {
                        "montage_svg": artifact.montage_svg.as_posix(),
                        "animation_svg": artifact.animation_svg.as_posix(),
                    }
                    for artifact in artifacts
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
