"""Validate a SCHUNK Isaac render wrapper run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.render_validation import validate_render_run_outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate fresh SCHUNK Isaac render outputs.")
    parser.add_argument("--render-dir", required=True, type=Path)
    parser.add_argument("--run-start-epoch", required=True, type=float)
    parser.add_argument(
        "--required-visual-evidence",
        choices=("schunk_mesh", "schunk_mesh_with_link_skeleton", "procedural_visual_rig", "articulation_only"),
        default="schunk_mesh",
        help="Evidence source required for visual RPS distinction.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = validate_render_run_outputs(
            args.render_dir,
            run_start_epoch=args.run_start_epoch,
            required_visual_evidence=args.required_visual_evidence,
        )
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
