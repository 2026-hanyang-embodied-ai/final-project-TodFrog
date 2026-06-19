"""Create the v4 calibration recording folder scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.v4_calibration_scaffold import V4CalibrationScaffoldConfig, prepare_v4_calibration_scaffold


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create non-held-out v4 calibration recording folders.")
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--expected-per-label", type=int, default=20)
    args = parser.parse_args(argv)

    summary = prepare_v4_calibration_scaffold(
        V4CalibrationScaffoldConfig(
            calibration_root=args.calibration_root,
            heldout_roots=tuple(args.heldout_root),
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
