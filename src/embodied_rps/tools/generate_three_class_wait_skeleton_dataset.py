"""Generate the 3-class paper-wait skeleton dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.three_class_wait_skeletons import (
    ThreeClassWaitExpansionConfig,
    generate_three_class_wait_dataset,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate 3-class shards for realtime paper-wait prediction."""

    parser = argparse.ArgumentParser(description="Generate 3-class paper-wait skeleton shards.")
    parser.add_argument(
        "--base-dataset-root",
        type=Path,
        default=Path("artifacts/real_guided_large_sharded_20260610"),
        help="Existing non-test binary skeleton shard dataset to preserve and augment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_guided_three_class_wait_expanded_v2_20260611"),
        help="Output artifact root.",
    )
    parser.add_argument("--generated-per-target", type=int, default=10000, help="Procedural samples per class.")
    parser.add_argument("--base-rock-stride", type=int, default=2, help="Use every Nth base sample as an early-fist rock hold.")
    parser.add_argument("--sequence-length", type=int, default=72, help="Padded sequence length.")
    parser.add_argument("--min-length", type=int, default=48, help="Minimum generated valid sequence length.")
    parser.add_argument("--shard-size", type=int, default=512, help="Output shard size.")
    parser.add_argument("--seed", type=int, default=20260611, help="Deterministic generation seed.")
    parser.add_argument(
        "--augmentation-profile",
        choices=(
            "baseline",
            "v2_targeted",
            "v3_targeted",
            "v4_fewshot",
            "v4_contrastive",
            "v4_rebalanced",
            "v4_failure_focused",
            "v4_remaining_gate",
            "v4_selector_targets",
            "v4_temporal_curl",
            "v4_boundary_pairs",
            "v4_hard_paper_scissors",
            "v4_delayed_paper_timing",
            "v4_mixed_paper_timing",
            "v4_live_prompt_hard",
            "v4_final_gate_micro",
            "v4_paper_rescue_micro",
            "v4_prompt_wait_hard",
            "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue",
            "v7d_real_seeded_prompt_window_guard",
            "v7e_stage1_paper_transition_rescue",
        ),
        default="v2_targeted",
        help="Procedural augmentation profile.",
    )
    parser.add_argument(
        "--calibration-seed-package-root",
        type=Path,
        default=None,
        help="Optional approved v4 calibration seed package root containing v4_calibration_seed_dataset.npz.",
    )
    parser.add_argument(
        "--live-rock-seed-package-root",
        type=Path,
        default=None,
        help="Optional overlay-derived live rock false-trigger seed package root.",
    )
    parser.add_argument(
        "--v7-seed-package-root",
        type=Path,
        default=None,
        help="Optional manually approved v7 RPS seed package root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files under output root.")
    args = parser.parse_args(argv)

    config = ThreeClassWaitExpansionConfig(
        output_root=args.output_root,
        base_dataset_root=args.base_dataset_root,
        generated_per_target=int(args.generated_per_target),
        sequence_length=int(args.sequence_length),
        min_length=int(args.min_length),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        base_rock_stride=int(args.base_rock_stride),
        augmentation_profile=args.augmentation_profile,
        calibration_seed_package_root=args.calibration_seed_package_root,
        live_rock_seed_package_root=args.live_rock_seed_package_root,
        v7_seed_package_root=args.v7_seed_package_root,
        overwrite=bool(args.overwrite),
    )
    summary = generate_three_class_wait_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
