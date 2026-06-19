"""Write the v7e stage1 local smoke preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
)
from embodied_rps.v7e_stage1_local_smoke_preflight import (
    DEFAULT_OUTPUT_ROOT,
    V7EStage1LocalSmokePreflightConfig,
    write_v7e_stage1_local_smoke_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7e stage1 local smoke preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-dataset-root", type=Path, default=DEFAULT_STAGE1_DATASET_ROOT)
    parser.add_argument("--stage1-training-config", type=Path, default=DEFAULT_STAGE1_TRAINING_CONFIG)
    args = parser.parse_args(argv)

    summary = write_v7e_stage1_local_smoke_preflight(
        V7EStage1LocalSmokePreflightConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            stage1_dataset_root=args.stage1_dataset_root,
            stage1_training_config=args.stage1_training_config,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_stage1_local_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
