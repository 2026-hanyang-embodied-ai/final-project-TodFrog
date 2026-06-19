"""CLI for sweeping open-set guards on saved real-skeleton validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_rps.real_skeleton_open_set_guard import sweep_open_set_guard_policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--confidence-thresholds",
        default="0.70",
        help="Comma-separated confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--margin-thresholds",
        default="0.10",
        help="Comma-separated confidence-margin thresholds to evaluate.",
    )
    parser.add_argument(
        "--confirmation-counts",
        default="2,3",
        help="Comma-separated rolling confirmation counts to evaluate.",
    )
    parser.add_argument(
        "--max-decision-progress-values",
        default="0.50",
        help="Comma-separated decision-progress deadlines to evaluate.",
    )
    parser.add_argument(
        "--transition-mass-thresholds",
        default="0.05",
        help="Comma-separated P(paper)+P(scissors) wait thresholds to evaluate.",
    )
    parser.add_argument(
        "--paper-wait-terminal-for-transitions",
        default="false",
        help="Comma-separated booleans. false means wait_counter_paper is provisional for paper/scissors clips.",
    )
    parser.add_argument(
        "--binary-transition-mass-thresholds",
        default="0.60",
        help="Comma-separated minimum P(paper)+P(scissors) required before binary decisions.",
    )
    parser.add_argument(
        "--min-binary-decision-progress-values",
        default="0.00,0.025,0.05,0.075,0.10,0.15",
        help="Comma-separated progress values before which binary spikes become provisional wait.",
    )
    parser.add_argument(
        "--early-binary-actions",
        default="wait_counter_paper",
        help="Comma-separated actions for early binary spikes: wait_counter_paper or suppress.",
    )
    args = parser.parse_args()

    result = sweep_open_set_guard_policies(
        validation_root=args.validation_root,
        output_root=args.output_root,
        confidence_thresholds=_parse_floats(args.confidence_thresholds),
        margin_thresholds=_parse_floats(args.margin_thresholds),
        confirmation_counts=_parse_ints(args.confirmation_counts),
        max_decision_progress_values=_parse_floats(args.max_decision_progress_values),
        transition_mass_thresholds=_parse_floats(args.transition_mass_thresholds),
        paper_wait_terminal_for_transition_values=_parse_bools(args.paper_wait_terminal_for_transitions),
        binary_transition_mass_thresholds=_parse_floats(args.binary_transition_mass_thresholds),
        min_binary_decision_progress_values=_parse_floats(args.min_binary_decision_progress_values),
        early_binary_actions=_parse_strings(args.early_binary_actions),
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
                "open_set_guard_sweep_summary_path": result["open_set_guard_sweep_summary_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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
