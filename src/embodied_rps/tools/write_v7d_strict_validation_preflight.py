"""Write the v7d strict validation and promotion preflight artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_strict_validation_preflight import (
    V7DStrictValidationPreflightConfig,
    write_v7d_strict_validation_preflight,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a fail-closed v7d strict validation preflight.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=V7DStrictValidationPreflightConfig.output_root)
    parser.add_argument(
        "--remote-training-preflight-root",
        type=Path,
        default=V7DStrictValidationPreflightConfig.remote_training_preflight_root,
    )
    parser.add_argument("--stage1-profile-json", type=Path, default=V7DStrictValidationPreflightConfig.stage1_profile_json)
    parser.add_argument("--stage2-profile-json", type=Path, default=V7DStrictValidationPreflightConfig.stage2_profile_json)
    parser.add_argument("--synthetic-metrics-root", type=Path, default=V7DStrictValidationPreflightConfig.synthetic_metrics_root)
    parser.add_argument(
        "--original20-validation-root",
        type=Path,
        default=V7DStrictValidationPreflightConfig.original20_validation_root,
    )
    parser.add_argument(
        "--heldout15-validation-root",
        type=Path,
        default=V7DStrictValidationPreflightConfig.heldout15_validation_root,
    )
    parser.add_argument("--replay-diagnostics-root", type=Path, default=V7DStrictValidationPreflightConfig.replay_diagnostics_root)
    parser.add_argument("--fresh-live-root", type=Path, default=V7DStrictValidationPreflightConfig.fresh_live_root)
    parser.add_argument("--event-manifest", type=Path, default=V7DStrictValidationPreflightConfig.event_manifest_path)
    parser.add_argument("--dataset-search-root", type=Path, default=V7DStrictValidationPreflightConfig.dataset_search_root)
    args = parser.parse_args(argv)

    summary = write_v7d_strict_validation_preflight(
        V7DStrictValidationPreflightConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            remote_training_preflight_root=args.remote_training_preflight_root,
            stage1_profile_json=args.stage1_profile_json,
            stage2_profile_json=args.stage2_profile_json,
            synthetic_metrics_root=args.synthetic_metrics_root,
            original20_validation_root=args.original20_validation_root,
            heldout15_validation_root=args.heldout15_validation_root,
            replay_diagnostics_root=args.replay_diagnostics_root,
            fresh_live_root=args.fresh_live_root,
            event_manifest_path=args.event_manifest,
            dataset_search_root=args.dataset_search_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "v7d_promotion_candidate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
