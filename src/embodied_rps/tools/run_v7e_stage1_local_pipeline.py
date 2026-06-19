"""Run or dry-run the v7e stage1 local data pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT,
    DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_HANDOFF_OUTPUT_ROOT,
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
    DEFAULT_THREE_CLASS_DATASET_ROOT,
)
from embodied_rps.v7e_stage1_local_smoke_preflight import DEFAULT_OUTPUT_ROOT as DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT
from embodied_rps.v7e_seed_package_preflight import DEFAULT_V7E_SEED_PACKAGE_ROOT
from embodied_rps.v7e_stage1_local_pipeline import (
    DEFAULT_OUTPUT_ROOT,
    V7EStage1LocalPipelineConfig,
    run_v7e_stage1_local_pipeline,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the fail-closed v7e stage1 local data pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed-package-root", type=Path, default=DEFAULT_V7E_SEED_PACKAGE_ROOT)
    parser.add_argument("--three-class-dataset-root", type=Path, default=DEFAULT_THREE_CLASS_DATASET_ROOT)
    parser.add_argument("--stage1-dataset-root", type=Path, default=DEFAULT_STAGE1_DATASET_ROOT)
    parser.add_argument("--base-dataset-root", type=Path, default=DEFAULT_BASE_DATASET_ROOT)
    parser.add_argument("--calibration-seed-package-root", type=Path, default=DEFAULT_CALIBRATION_SEED_PACKAGE_ROOT)
    parser.add_argument("--live-rock-seed-package-root", type=Path, default=DEFAULT_LIVE_ROCK_SEED_PACKAGE_ROOT)
    parser.add_argument("--stage1-training-config", type=Path, default=DEFAULT_STAGE1_TRAINING_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--handoff-output-root", type=Path, default=DEFAULT_HANDOFF_OUTPUT_ROOT)
    parser.add_argument("--local-smoke-preflight-root", type=Path, default=DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT)
    parser.add_argument("--generated-per-target", type=int, default=10000)
    parser.add_argument("--sequence-length", type=int, default=72)
    parser.add_argument("--min-length", type=int, default=48)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--base-rock-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--execute-local", action="store_true")
    parser.add_argument("--overwrite-outputs", action="store_true")
    args = parser.parse_args(argv)

    summary = run_v7e_stage1_local_pipeline(
        V7EStage1LocalPipelineConfig(
            project_root=args.project_root,
            seed_package_root=args.seed_package_root,
            three_class_dataset_root=args.three_class_dataset_root,
            stage1_dataset_root=args.stage1_dataset_root,
            base_dataset_root=args.base_dataset_root,
            calibration_seed_package_root=args.calibration_seed_package_root,
            live_rock_seed_package_root=args.live_rock_seed_package_root,
            stage1_training_config=args.stage1_training_config,
            output_root=args.output_root,
            handoff_output_root=args.handoff_output_root,
            local_smoke_preflight_root=args.local_smoke_preflight_root,
            generated_per_target=int(args.generated_per_target),
            sequence_length=int(args.sequence_length),
            min_length=int(args.min_length),
            shard_size=int(args.shard_size),
            base_rock_stride=int(args.base_rock_stride),
            seed=int(args.seed),
            execute_local=bool(args.execute_local),
            overwrite_outputs=bool(args.overwrite_outputs),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    status = str(summary.get("status", ""))
    if status.startswith("blocked_") or status == "official_full_scale_required":
        return 2
    if status.endswith("_failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
