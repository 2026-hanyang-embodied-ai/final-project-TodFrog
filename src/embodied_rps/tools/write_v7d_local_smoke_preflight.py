"""Write the v7d local smoke preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_local_smoke_preflight import (
    V7DLocalSmokePreflightConfig,
    write_v7d_local_smoke_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7d local smoke preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=V7DLocalSmokePreflightConfig.output_root)
    parser.add_argument("--stage1-dataset-root", type=Path, default=V7DLocalSmokePreflightConfig.stage1_dataset_root)
    parser.add_argument("--stage2-dataset-root", type=Path, default=V7DLocalSmokePreflightConfig.stage2_dataset_root)
    parser.add_argument("--stage1-training-config", type=Path, default=V7DLocalSmokePreflightConfig.stage1_training_config)
    parser.add_argument("--stage2-training-config", type=Path, default=V7DLocalSmokePreflightConfig.stage2_training_config)
    args = parser.parse_args(argv)

    summary = write_v7d_local_smoke_preflight(
        V7DLocalSmokePreflightConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            stage1_dataset_root=args.stage1_dataset_root,
            stage2_dataset_root=args.stage2_dataset_root,
            stage1_training_config=args.stage1_training_config,
            stage2_training_config=args.stage2_training_config,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "ready_for_local_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
