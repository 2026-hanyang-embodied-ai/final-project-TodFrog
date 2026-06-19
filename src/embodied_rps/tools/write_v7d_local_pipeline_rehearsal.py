"""Write the sandbox v7d local pipeline rehearsal artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_local_pipeline_rehearsal import (
    V7DLocalPipelineRehearsalConfig,
    write_v7d_local_pipeline_rehearsal,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a sandbox-only v7d local post-approval pipeline rehearsal.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-root", type=Path, default=V7DLocalPipelineRehearsalConfig.review_root)
    parser.add_argument("--shortlist-root", type=Path, default=V7DLocalPipelineRehearsalConfig.shortlist_root)
    parser.add_argument("--selection-root", type=Path, default=V7DLocalPipelineRehearsalConfig.selection_root)
    parser.add_argument("--prefill-root", type=Path, default=V7DLocalPipelineRehearsalConfig.prefill_root)
    parser.add_argument("--output-root", type=Path, default=V7DLocalPipelineRehearsalConfig.output_root)
    parser.add_argument("--generated-per-target", type=int, default=V7DLocalPipelineRehearsalConfig.generated_per_target)
    parser.add_argument("--shard-size", type=int, default=V7DLocalPipelineRehearsalConfig.shard_size)
    args = parser.parse_args(argv)

    summary = write_v7d_local_pipeline_rehearsal(
        V7DLocalPipelineRehearsalConfig(
            project_root=args.project_root,
            review_root=args.review_root,
            shortlist_root=args.shortlist_root,
            selection_root=args.selection_root,
            prefill_root=args.prefill_root,
            output_root=args.output_root,
            generated_per_target=int(args.generated_per_target),
            shard_size=int(args.shard_size),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "sandbox_local_v7d_datasets_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
