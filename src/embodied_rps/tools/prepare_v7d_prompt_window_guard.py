"""Write v7d two-stage prompt-window guard preparation artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_preparation import (
    DEFAULT_CANDIDATE_ROOTS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_V7B_AUDIT_ROOT,
    DEFAULT_V7C_AUDIT_ROOT,
    DEFAULT_V7C_GATE_ROOT,
    V7DPreparationConfig,
    prepare_v7d_prompt_window_guard,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare v7d two-stage prompt-window guard artifacts without training.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--v7b-audit-root", type=Path, default=DEFAULT_V7B_AUDIT_ROOT)
    parser.add_argument("--v7c-audit-root", type=Path, default=DEFAULT_V7C_AUDIT_ROOT)
    parser.add_argument("--v7c-gate-root", type=Path, default=DEFAULT_V7C_GATE_ROOT)
    parser.add_argument("--candidate-root", type=Path, action="append", default=None)
    parser.add_argument("--dataset-search-root", type=Path, default=Path("D:/dataset"))
    args = parser.parse_args(argv)

    candidate_roots = tuple(args.candidate_root) if args.candidate_root is not None else DEFAULT_CANDIDATE_ROOTS
    summary = prepare_v7d_prompt_window_guard(
        V7DPreparationConfig(
            project_root=args.project_root,
            output_root=args.output_root,
            v7b_audit_root=args.v7b_audit_root,
            v7c_audit_root=args.v7c_audit_root,
            v7c_gate_root=args.v7c_gate_root,
            candidate_roots=candidate_roots,
            dataset_search_root=args.dataset_search_root,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
