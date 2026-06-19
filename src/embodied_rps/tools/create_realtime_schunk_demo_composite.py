"""Create a final-demo composite video from realtime overlay and SCHUNK preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_rps.realtime_schunk_demo_composite import create_realtime_schunk_demo_composite


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose realtime skeleton overlay with SCHUNK response preview.")
    parser.add_argument("--overlay-video", required=True, type=Path, help="Realtime skeleton overlay MP4.")
    parser.add_argument("--response-preview-image", required=True, type=Path, help="SCHUNK response preview contact-sheet PNG.")
    parser.add_argument("--output-root", required=True, type=Path, help="Output directory.")
    parser.add_argument("--width", type=int, default=1920, help="Output video width.")
    parser.add_argument("--height", type=int, default=1080, help="Output video height.")
    args = parser.parse_args()

    manifest = create_realtime_schunk_demo_composite(
        overlay_video=args.overlay_video,
        response_preview_image=args.response_preview_image,
        out_dir=args.output_root,
        output_size=(int(args.width), int(args.height)),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
