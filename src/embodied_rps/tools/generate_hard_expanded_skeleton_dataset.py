"""Generate the view-robust hard-expanded skeleton dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.hard_example_skeletons import HardExpansionConfig, generate_hard_expanded_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Generate hard-expanded shards for real skeleton final-gesture training."""

    parser = argparse.ArgumentParser(description="Generate view-robust hard-example skeleton shards.")
    parser.add_argument(
        "--base-dataset-root",
        type=Path,
        default=Path("artifacts/real_guided_large_sharded_20260610"),
        help="Existing real-guided shard dataset to preserve and augment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/real_guided_hard_expanded_20260611"),
        help="Output artifact root.",
    )
    parser.add_argument("--generated-per-target", type=int, default=2500, help="Hard/generated samples per final gesture.")
    parser.add_argument("--sequence-length", type=int, default=72, help="Padded sequence length.")
    parser.add_argument("--min-length", type=int, default=48, help="Minimum generated valid sequence length.")
    parser.add_argument("--shard-size", type=int, default=512, help="Output shard size.")
    parser.add_argument("--seed", type=int, default=20260611, help="Deterministic generation seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files under output root.")
    parser.add_argument(
        "--no-base",
        action="store_true",
        help="Generate only procedural hard examples instead of preserving the base dataset.",
    )
    args = parser.parse_args(argv)

    config = HardExpansionConfig(
        output_root=args.output_root,
        base_dataset_root=None if args.no_base else args.base_dataset_root,
        generated_per_target=int(args.generated_per_target),
        sequence_length=int(args.sequence_length),
        min_length=int(args.min_length),
        shard_size=int(args.shard_size),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
    )
    summary = generate_hard_expanded_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
