"""Create SCHUNK visual evidence montage figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.visual_evidence_montage import create_schunk_visual_evidence_montages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create SCHUNK procedural visual-rig montage figures.")
    parser.add_argument("--render-dir", type=Path, default=Path("results/isaac_schunk_visual_rig_renders"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/schunk_visual_evidence_figures"))
    parser.add_argument("--yaw", type=int, default=45)
    parser.add_argument("--pitch", type=int, default=20)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--cell-height", type=int, default=180)
    args = parser.parse_args(argv)

    artifacts = create_schunk_visual_evidence_montages(
        render_dir=args.render_dir,
        out_dir=args.out_dir,
        yaw=args.yaw,
        pitch=args.pitch,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "static_montage": artifacts.static_montage.as_posix(),
                "sequence_montage": artifacts.sequence_montage.as_posix(),
                "manifest": artifacts.manifest.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
