"""Build the realtime demo submission candidate packet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_submission_packet import (
    RealtimeDemoSubmissionPacketConfig,
    build_realtime_demo_submission_packet,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for submission candidate packet generation."""

    parser = argparse.ArgumentParser(description="Build a realtime demo submission candidate packet.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoSubmissionPacketConfig.output_root)
    parser.add_argument("--final-candidate", type=Path, default=RealtimeDemoSubmissionPacketConfig.final_candidate)
    args = parser.parse_args(argv)

    summary = build_realtime_demo_submission_packet(
        RealtimeDemoSubmissionPacketConfig(
            output_root=args.output_root,
            final_candidate=args.final_candidate,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
