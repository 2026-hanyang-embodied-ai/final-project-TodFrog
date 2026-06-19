"""Write the non-mutating v7e approval readiness handoff."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7e_approval_readiness_handoff import (
    V7EApprovalReadinessHandoffConfig,
    write_v7e_approval_readiness_handoff,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write v7e approval readiness handoff.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan-root", type=Path, default=V7EApprovalReadinessHandoffConfig.plan_root)
    parser.add_argument("--validator-root", type=Path, default=V7EApprovalReadinessHandoffConfig.validator_root)
    parser.add_argument("--applier-root", type=Path, default=V7EApprovalReadinessHandoffConfig.applier_root)
    parser.add_argument("--output-root", type=Path, default=V7EApprovalReadinessHandoffConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7e_approval_readiness_handoff(
        V7EApprovalReadinessHandoffConfig(
            project_root=args.project_root,
            plan_root=args.plan_root,
            validator_root=args.validator_root,
            applier_root=args.applier_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {
        "awaiting_v7e_paper_seed_approval",
        "ready_for_v7e_official_seed_package_sequence",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
