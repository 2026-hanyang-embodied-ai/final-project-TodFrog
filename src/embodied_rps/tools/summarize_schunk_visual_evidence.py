"""Summarize SCHUNK render evidence modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.render_evidence import summarize_schunk_visual_evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize SCHUNK mesh and procedural visual-rig render evidence.")
    parser.add_argument("--schunk-render-dir", type=Path, default=Path("results/isaac_schunk_renders"))
    parser.add_argument("--visual-rig-render-dir", type=Path, default=Path("results/isaac_schunk_visual_rig_renders"))
    parser.add_argument("--out-json", type=Path, default=Path("results/schunk_visual_evidence_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/schunk_visual_evidence_summary.md"))
    args = parser.parse_args(argv)

    summary = summarize_schunk_visual_evidence(
        schunk_render_dir=args.schunk_render_dir,
        visual_rig_render_dir=args.visual_rig_render_dir,
        out_json=args.out_json,
        out_markdown=args.out_md,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
