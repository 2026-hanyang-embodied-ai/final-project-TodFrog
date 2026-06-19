"""Write the v7d post-validation failure audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_post_validation_failure_audit import (
    V7DPostValidationFailureAuditConfig,
    write_v7d_post_validation_failure_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a v7d post-validation failure audit.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--original20-validation-root",
        type=Path,
        default=V7DPostValidationFailureAuditConfig.original20_validation_root,
    )
    parser.add_argument("--output-root", type=Path, default=V7DPostValidationFailureAuditConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_post_validation_failure_audit(
        V7DPostValidationFailureAuditConfig(
            project_root=args.project_root,
            original20_validation_root=args.original20_validation_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
