"""Create the raw v4 recording staging folder scaffold."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_staging_scaffold import (
    V4RecordingStagingScaffoldConfig,
    prepare_v4_recording_staging_scaffold,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Create label folders and guidance for raw v4 staging MP4s."""

    parser = argparse.ArgumentParser(description="Create raw non-held-out v4 recording staging folders.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_recording_staging")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", action="append", type=Path, default=[])
    parser.add_argument("--expected-per-label", type=int, default=20)
    args = parser.parse_args(argv)

    summary = prepare_v4_recording_staging_scaffold(
        V4RecordingStagingScaffoldConfig(
            staging_root=args.staging_root,
            calibration_root=args.calibration_root,
            heldout_roots=tuple(args.heldout_root or [DEFAULT_LOCAL_DATA_ROOT / "test"]),
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
