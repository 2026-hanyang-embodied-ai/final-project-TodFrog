"""Select the final realtime demo video candidate from archived run summaries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.realtime_demo_final_candidate import (
    RealtimeDemoFinalCandidateConfig,
    select_realtime_demo_final_candidate,
)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for final realtime demo candidate selection."""

    parser = argparse.ArgumentParser(description="Select a final realtime demo video candidate.")
    parser.add_argument("--output-root", type=Path, default=RealtimeDemoFinalCandidateConfig.output_root)
    parser.add_argument("--archive-index", type=Path, default=RealtimeDemoFinalCandidateConfig.archive_index)
    args = parser.parse_args(argv)

    summary = select_realtime_demo_final_candidate(
        RealtimeDemoFinalCandidateConfig(
            output_root=args.output_root,
            archive_index=args.archive_index,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
