"""CLI for final README/report/notebook support assets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.submission_assets import build_final_submission_assets, write_metrics_csv


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final submission documentation support artifacts.")
    parser.add_argument("--docs-root", type=Path, default=Path("docs"), help="Documentation output root.")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("conference-latex-template/IEEE-conference-template-062824"),
        help="IEEE report template root.",
    )
    parser.add_argument("--tcn-image", type=Path, default=None, help="Optional user-provided TCN diagram to copy.")
    args = parser.parse_args(argv)

    project_root = Path.cwd()
    summary = build_final_submission_assets(
        project_root=project_root,
        docs_root=args.docs_root,
        report_root=args.report_root,
        tcn_image=args.tcn_image,
    )
    write_metrics_csv(project_root / args.docs_root / "final_submission_metrics.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
