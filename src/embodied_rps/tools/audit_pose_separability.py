"""Audit robust RPS pose-family semantic separability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.pose_family import audit_pose_family_dataset, load_pose_family_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Audit an existing pose-family dataset."""

    parser = argparse.ArgumentParser(description="Audit robust RPS pose-family dataset separability.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to data/synthetic/rps_pose_family_dataset.npz")
    parser.add_argument("--out", default=Path("results/pose_family_audit.json"), type=Path, help="Output audit JSON path.")
    args = parser.parse_args(argv)

    dataset = load_pose_family_dataset(args.dataset)
    audit = audit_pose_family_dataset(dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
