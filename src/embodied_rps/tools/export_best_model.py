"""Export a configured model run as a loadable inference profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.checkpoints import export_model_profile


def main(argv: Sequence[str] | None = None) -> int:
    """Retrain one configured run and save its checkpoint profile."""

    parser = argparse.ArgumentParser(description="Export a trained RPS classifier profile.")
    parser.add_argument("--config", required=True, type=Path, help="Path to configs/model_sweep.yaml")
    parser.add_argument("--run-id", required=True, help="Stable model run id to export.")
    parser.add_argument("--profile", required=True, help="Named profile, e.g. gru_clear50.")
    args = parser.parse_args(argv)

    metadata_path = export_model_profile(config_path=args.config, run_id=str(args.run_id), profile=str(args.profile))
    print(json.dumps({"profile": str(args.profile), "metadata_path": metadata_path.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
