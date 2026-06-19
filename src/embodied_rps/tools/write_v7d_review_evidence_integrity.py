"""Write the v7d required-role review evidence integrity audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.v7d_review_evidence_integrity import (
    V7DReviewEvidenceIntegrityConfig,
    write_v7d_review_evidence_integrity,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit v7d required-role manual-review evidence files.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--packet-root", type=Path, default=V7DReviewEvidenceIntegrityConfig.packet_root)
    parser.add_argument("--output-root", type=Path, default=V7DReviewEvidenceIntegrityConfig.output_root)
    args = parser.parse_args(argv)

    summary = write_v7d_review_evidence_integrity(
        V7DReviewEvidenceIntegrityConfig(
            project_root=args.project_root,
            packet_root=args.packet_root,
            output_root=args.output_root,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "ready_for_manual_temporal_approval" else 2


if __name__ == "__main__":
    raise SystemExit(main())
