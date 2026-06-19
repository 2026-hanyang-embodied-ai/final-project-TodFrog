"""Write deterministic v4 calibration recording slots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import DEFAULT_LOCAL_DATA_ROOT
from embodied_rps.v4_recording_slot_manifest import V4RecordingSlotManifestConfig, write_v4_recording_slot_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Write the v4 recording slot manifest."""

    parser = argparse.ArgumentParser(description="Write v4 calibration recording slot targets.")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--expected-per-label", type=int, default=20)
    args = parser.parse_args(argv)

    summary = write_v4_recording_slot_manifest(
        V4RecordingSlotManifestConfig(
            calibration_root=args.calibration_root,
            expected_per_label=int(args.expected_per_label),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
