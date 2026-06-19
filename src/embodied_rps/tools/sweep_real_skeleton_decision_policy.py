"""CLI for sweeping saved real-skeleton video decision policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_rps.real_skeleton_policy_sweep import sweep_decision_policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--confidence-thresholds",
        default="0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--margin-thresholds",
        default="0.10,0.20,0.30,0.40",
        help="Comma-separated confidence-margin thresholds to evaluate.",
    )
    parser.add_argument(
        "--confirmation-counts",
        default="2,3,4,5",
        help="Comma-separated rolling confirmation counts to evaluate.",
    )
    parser.add_argument(
        "--max-decision-progress-values",
        default="0.50,0.60,0.75",
        help="Comma-separated decision-progress deadlines to evaluate.",
    )
    parser.add_argument(
        "--transition-mass-thresholds",
        default="0.05,0.10,0.15,0.25,0.35",
        help="Comma-separated P(paper)+P(scissors) wait thresholds to evaluate.",
    )
    parser.add_argument(
        "--paper-wait-terminal-for-transitions",
        default="true,false",
        help="Comma-separated booleans. false means wait_counter_paper is provisional for paper/scissors clips.",
    )
    parser.add_argument(
        "--binary-transition-mass-thresholds",
        default="0.0",
        help="Comma-separated minimum P(paper)+P(scissors) values required before a binary paper/scissors decision.",
    )
    args = parser.parse_args()

    result = sweep_decision_policies(
        validation_root=args.validation_root,
        output_root=args.output_root,
        confidence_thresholds=_parse_floats(args.confidence_thresholds),
        margin_thresholds=_parse_floats(args.margin_thresholds),
        confirmation_counts=_parse_ints(args.confirmation_counts),
        max_decision_progress_values=_parse_floats(args.max_decision_progress_values),
        transition_mass_thresholds=_parse_floats(args.transition_mass_thresholds),
        paper_wait_terminal_for_transition_values=_parse_bools(args.paper_wait_terminal_for_transitions),
        binary_transition_mass_thresholds=_parse_floats(args.binary_transition_mass_thresholds),
    )
    print(
        json.dumps(
            {
                "policy_count": result["policy_count"],
                "clip_count": result["clip_count"],
                "best_policy": result["best_policy"],
                "best_summary": {
                    "passed": result["best_summary"]["passed"],
                    "passed_clip_count": result["best_summary"]["passed_clip_count"],
                    "failed_clip_count": result["best_summary"]["failed_clip_count"],
                    "paper_scissors_accuracy": result["best_summary"]["paper_scissors_accuracy"],
                    "rock_wait_success_count": result["best_summary"]["rock_wait_success_count"],
                    "rock_false_trigger_count": result["best_summary"]["rock_false_trigger_count"],
                    "failure_reason_counts": result["best_summary"]["failure_reason_counts"],
                },
                "policy_sweep_summary_path": result["policy_sweep_summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_bools(value: str) -> list[bool]:
    parsed: list[bool] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "yes", "y"}:
            parsed.append(True)
        elif token in {"0", "false", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Expected boolean token, got: {part}")
    return parsed


if __name__ == "__main__":
    main()
