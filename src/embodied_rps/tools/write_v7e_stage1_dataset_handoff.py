"""Write the v7e stage1 dataset handoff summary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
    DEFAULT_THREE_CLASS_DATASET_ROOT,
    V7EStage1DatasetHandoffConfig,
    write_v7e_stage1_dataset_handoff,
)
from embodied_rps.v7e_seed_package_preflight import DEFAULT_V7E_SEED_PACKAGE_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7e stage1 dataset handoff status without training.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed-package-root", type=Path, default=DEFAULT_V7E_SEED_PACKAGE_ROOT)
    parser.add_argument("--three-class-dataset-root", type=Path, default=DEFAULT_THREE_CLASS_DATASET_ROOT)
    parser.add_argument("--stage1-dataset-root", type=Path, default=DEFAULT_STAGE1_DATASET_ROOT)
    parser.add_argument("--stage1-training-config", type=Path, default=DEFAULT_STAGE1_TRAINING_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    summary = write_v7e_stage1_dataset_handoff(
        V7EStage1DatasetHandoffConfig(
            project_root=args.project_root,
            seed_package_root=args.seed_package_root,
            three_class_dataset_root=args.three_class_dataset_root,
            stage1_dataset_root=args.stage1_dataset_root,
            stage1_training_config=args.stage1_training_config,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    return 2 if status.startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
