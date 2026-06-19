"""CLI for saved-output conditional scissors-rescue selector sweeps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embodied_rps.real_skeleton_scissors_rescue_sweep import sweep_scissors_rescue_selectors
from embodied_rps.real_skeleton_video_eval import StrictDecisionConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep conditional scissors rescue over saved baseline and candidate validation roots."
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--candidate-confidence-thresholds", default="0.90,0.95,0.98,0.99")
    parser.add_argument("--candidate-margin-thresholds", default="0.80,0.90,0.95,0.98")
    parser.add_argument("--baseline-transition-mass-thresholds", default="0.20,0.40,0.60,0.80")
    parser.add_argument("--baseline-rock-max-values", default="0.20,0.40,0.60")
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--confirmation-count", type=int, default=3)
    parser.add_argument("--max-decision-progress", type=float, default=0.50)
    parser.add_argument("--transition-mass-threshold", type=float, default=0.15)
    parser.add_argument("--binary-transition-mass-threshold", type=float, default=0.0)
    parser.add_argument(
        "--min-binary-decision-progress",
        type=float,
        default=0.0,
        help="Treat binary paper/scissors decisions before this progress as provisional wait.",
    )
    parser.add_argument(
        "--paper-wait-nonterminal-for-transitions",
        action="store_true",
        help="Allow paper/scissors transition clips to continue after early wait_counter_paper frames.",
    )
    args = parser.parse_args(argv)

    decision_config = StrictDecisionConfig(
        confidence_threshold=float(args.confidence_threshold),
        margin_threshold=float(args.margin_threshold),
        confirmation_count=int(args.confirmation_count),
        max_decision_progress=float(args.max_decision_progress),
        transition_mass_threshold=float(args.transition_mass_threshold),
        paper_wait_is_terminal_for_transitions=not bool(args.paper_wait_nonterminal_for_transitions),
        binary_transition_mass_threshold=float(args.binary_transition_mass_threshold),
    )
    result = sweep_scissors_rescue_selectors(
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
        output_root=args.output_root,
        candidate_confidence_thresholds=_parse_float_list(args.candidate_confidence_thresholds),
        candidate_margin_thresholds=_parse_float_list(args.candidate_margin_thresholds),
        baseline_transition_mass_thresholds=_parse_float_list(args.baseline_transition_mass_thresholds),
        baseline_rock_max_values=_parse_optional_float_list(args.baseline_rock_max_values),
        decision_config=decision_config,
        min_binary_decision_progress=float(args.min_binary_decision_progress),
    )
    print(
        json.dumps(
            {
                "policy_count": result["policy_count"],
                "best_policy": result["best_policy"],
                "best_summary": result["best_summary"],
                "summary_path": result.get("scissors_rescue_sweep_summary_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric value")
    return parsed


def _parse_optional_float_list(value: str) -> list[float | None]:
    parsed: list[float | None] = []
    for part in value.split(","):
        stripped = part.strip().lower()
        if not stripped:
            continue
        if stripped in {"none", "null"}:
            parsed.append(None)
        else:
            parsed.append(float(stripped))
    if not parsed:
        raise ValueError("Expected at least one numeric or none value")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
