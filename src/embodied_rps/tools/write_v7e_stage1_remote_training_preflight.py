"""Write the v7e stage1 remote training preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_stage1_dataset_handoff import (
    DEFAULT_STAGE1_DATASET_ROOT,
    DEFAULT_STAGE1_TRAINING_CONFIG,
    DEFAULT_THREE_CLASS_DATASET_ROOT,
)
from embodied_rps.v7e_stage1_local_smoke_preflight import DEFAULT_OUTPUT_ROOT as DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT
from embodied_rps.v7e_stage1_remote_training_preflight import (
    DEFAULT_OUTPUT_ROOT,
    REMOTE_HOST,
    REMOTE_WORKSPACE,
    V7EStage1RemoteTrainingPreflightConfig,
    write_v7e_stage1_remote_training_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7e stage1 remote training preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--local-smoke-preflight-root", type=Path, default=DEFAULT_LOCAL_SMOKE_PREFLIGHT_ROOT)
    parser.add_argument("--three-class-dataset-root", type=Path, default=DEFAULT_THREE_CLASS_DATASET_ROOT)
    parser.add_argument("--stage1-dataset-root", type=Path, default=DEFAULT_STAGE1_DATASET_ROOT)
    parser.add_argument("--stage1-training-config", type=Path, default=DEFAULT_STAGE1_TRAINING_CONFIG)
    parser.add_argument("--remote-host", default=REMOTE_HOST)
    parser.add_argument("--remote-workspace", default=REMOTE_WORKSPACE)
    args = parser.parse_args(argv)

    summary = write_v7e_stage1_remote_training_preflight(
        V7EStage1RemoteTrainingPreflightConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            local_smoke_preflight_root=args.local_smoke_preflight_root,
            three_class_dataset_root=args.three_class_dataset_root,
            stage1_dataset_root=args.stage1_dataset_root,
            stage1_training_config=args.stage1_training_config,
            remote_host=str(args.remote_host),
            remote_workspace=str(args.remote_workspace),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_remote_stage1_tcn_training" else 2


if __name__ == "__main__":
    raise SystemExit(main())
