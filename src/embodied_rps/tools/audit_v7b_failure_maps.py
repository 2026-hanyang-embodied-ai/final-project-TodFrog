"""Write the v7b pre-branch failure-map audit artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7b_failure_audit import build_v7b_failure_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v7 strict validation failure maps before the v7b correction branch.")
    parser.add_argument(
        "--original20-validation-root",
        type=Path,
        default=Path("artifacts/real_mp4_prediction_validation_original20_v7_rps_pose_20260617"),
    )
    parser.add_argument(
        "--heldout15-validation-root",
        type=Path,
        default=Path("artifacts/real_mp4_prediction_validation_heldout15_v7_rps_pose_20260617"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_skeleton_v7b_failure_audit_20260618"),
    )
    args = parser.parse_args(argv)
    summary = build_v7b_failure_audit(
        original20_validation_root=args.original20_validation_root,
        heldout15_validation_root=args.heldout15_validation_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
