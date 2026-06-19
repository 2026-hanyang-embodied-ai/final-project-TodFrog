"""Write the v4 skeleton-prediction operator runbook."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v4_operator_runbook import (
    DEFAULT_LOCAL_DATA_ROOT,
    V4OperatorRunbookConfig,
    write_v4_operator_runbook,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the current v4 operator runbook."""

    parser = argparse.ArgumentParser(description="Write an ordered v4 skeleton-prediction operator runbook.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/real_skeleton_v4_operator_runbook_20260612"))
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "v4_calibration")
    parser.add_argument("--heldout-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT / "test")
    parser.add_argument("--original20-root", type=Path, default=DEFAULT_LOCAL_DATA_ROOT)
    parser.add_argument("--expected-per-label", type=int, default=20)
    parser.add_argument("--end-to-end-summary", type=Path, default=Path("artifacts/real_skeleton_v4_end_to_end_20260612/end_to_end_summary.json"))
    args = parser.parse_args(argv)

    runbook = write_v4_operator_runbook(
        V4OperatorRunbookConfig(
            output_root=args.output_root,
            calibration_root=args.calibration_root,
            heldout_root=args.heldout_root,
            original20_root=args.original20_root,
            expected_per_label=int(args.expected_per_label),
            end_to_end_summary_path=args.end_to_end_summary,
        )
    )
    print(json.dumps(runbook, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
