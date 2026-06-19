"""Write the v7d remote training preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_remote_training_preflight import (
    REMOTE_HOST,
    REMOTE_WORKSPACE,
    V7DRemoteTrainingPreflightConfig,
    write_v7d_remote_training_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7d remote training preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=V7DRemoteTrainingPreflightConfig.output_root)
    parser.add_argument(
        "--local-smoke-preflight-root",
        type=Path,
        default=V7DRemoteTrainingPreflightConfig.local_smoke_preflight_root,
    )
    parser.add_argument(
        "--three-class-dataset-root",
        type=Path,
        default=V7DRemoteTrainingPreflightConfig.three_class_dataset_root,
    )
    parser.add_argument("--stage1-dataset-root", type=Path, default=V7DRemoteTrainingPreflightConfig.stage1_dataset_root)
    parser.add_argument("--stage2-dataset-root", type=Path, default=V7DRemoteTrainingPreflightConfig.stage2_dataset_root)
    parser.add_argument("--stage1-training-config", type=Path, default=V7DRemoteTrainingPreflightConfig.stage1_training_config)
    parser.add_argument("--stage2-training-config", type=Path, default=V7DRemoteTrainingPreflightConfig.stage2_training_config)
    parser.add_argument("--remote-host", default=REMOTE_HOST)
    parser.add_argument("--remote-workspace", default=REMOTE_WORKSPACE)
    args = parser.parse_args(argv)

    summary = write_v7d_remote_training_preflight(
        V7DRemoteTrainingPreflightConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            local_smoke_preflight_root=args.local_smoke_preflight_root,
            three_class_dataset_root=args.three_class_dataset_root,
            stage1_dataset_root=args.stage1_dataset_root,
            stage2_dataset_root=args.stage2_dataset_root,
            stage1_training_config=args.stage1_training_config,
            stage2_training_config=args.stage2_training_config,
            remote_host=str(args.remote_host),
            remote_workspace=str(args.remote_workspace),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_remote_tcn_training" else 2


if __name__ == "__main__":
    raise SystemExit(main())
