"""Train real skeleton final-gesture predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from embodied_rps.real_skeleton_training import (
    export_best_real_profile,
    load_sweep_config,
    train_real_model_runs,
    write_real_model_comparison,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Train one real-skeleton model family or all configured families."""

    parser = argparse.ArgumentParser(description="Train real skeleton final-gesture predictors.")
    parser.add_argument("--config", required=True, type=Path, help="Path to real skeleton predictor YAML.")
    parser.add_argument("--model", required=True, choices=("gru", "tcn", "transformer", "all"), help="Model family.")
    parser.add_argument("--smoke", action="store_true", help="Use smoke_epochs and write to a smoke run folder.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on expanded runs.")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip model comparison and realtime profile export.",
    )
    args = parser.parse_args(argv)

    config = load_sweep_config(args.config)
    completed = train_real_model_runs(
        sweep_config=config,
        requested_model=str(args.model),
        smoke=bool(args.smoke),
        max_runs=args.max_runs,
    )

    runs_dir = Path(_string_value(config, "runs_dir"))
    comparison_path = Path(_string_value(config, "comparison_path"))
    if args.smoke:
        runs_dir = runs_dir / "smoke"
        comparison_path = comparison_path.with_name("smoke_" + comparison_path.name)

    summary: dict[str, object] = {
        "completed_runs": len(completed),
        "run_ids": [str(run["run_id"]) for run in completed],
        "runs_dir": runs_dir.as_posix(),
    }
    if not args.skip_export:
        comparison = write_real_model_comparison(runs_dir, comparison_path)
        summary["comparison_path"] = comparison_path.as_posix()
        summary["model_ready"] = comparison["model_ready"]
        if not args.smoke:
            profile = export_best_real_profile(
                runs_dir=runs_dir,
                output_dir=Path(_string_value(config, "profile_dir")),
                profile_name=_string_value(config, "best_profile"),
                preferred_model=_preferred_export_model(config),
                ratio="0.50",
            )
            summary["profile_path"] = str(profile["model_state_path"])
            summary["profile_json"] = str(Path(_string_value(config, "profile_dir")) / f"{_string_value(config, 'best_profile')}.json")

    print(json.dumps(summary, indent=2))
    return 0


def _string_value(mapping: dict[str, object], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _preferred_export_model(mapping: dict[str, object]) -> str:
    value = mapping.get("preferred_export_model", "gru")
    if not isinstance(value, str) or value not in {"gru", "tcn", "transformer"}:
        raise ValueError("preferred_export_model must be one of: gru, tcn, transformer")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
