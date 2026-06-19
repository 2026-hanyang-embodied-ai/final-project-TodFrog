"""Preparation artifacts for the v7d two-stage prompt-window guard branch."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7d_prompt_window_guard_preparation_20260618")
DEFAULT_V7B_AUDIT_ROOT = Path("artifacts/real_skeleton_v7b_post_validation_failure_audit_20260618")
DEFAULT_V7C_AUDIT_ROOT = Path("artifacts/real_skeleton_v7c_post_validation_failure_audit_20260618")
DEFAULT_V7C_GATE_ROOT = Path("artifacts/real_skeleton_v7c_training_gate_20260618")
DEFAULT_CANDIDATE_ROOTS = (Path("artifacts/real_skeleton_v7_rps_seed_package_20260617"),)
REQUIRED_REAL_SEED_ROLES: tuple[str, ...] = (
    "hard_paper_prompt_window",
    "rock_wait_prompt_window",
    "scissors_boundary_control",
)
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "source_path",
    "source_overlay_video",
    "source_skeleton_npz",
    "source_frame_log",
    "skeleton_npz",
    "preview_image",
)


@dataclass(frozen=True)
class V7DPreparationConfig:
    """Inputs for writing v7d status-only preparation artifacts."""

    project_root: Path = field(default_factory=Path.cwd)
    output_root: Path = DEFAULT_OUTPUT_ROOT
    v7b_audit_root: Path = DEFAULT_V7B_AUDIT_ROOT
    v7c_audit_root: Path = DEFAULT_V7C_AUDIT_ROOT
    v7c_gate_root: Path = DEFAULT_V7C_GATE_ROOT
    candidate_roots: tuple[Path, ...] = DEFAULT_CANDIDATE_ROOTS
    dataset_search_root: Path = Path("D:/dataset")


def prepare_v7d_prompt_window_guard(config: V7DPreparationConfig) -> dict[str, object]:
    """Write v7d planning/preflight artifacts without generating a dataset or training."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    v7b_audit_root = _resolve_path(project_root, config.v7b_audit_root)
    v7c_audit_root = _resolve_path(project_root, config.v7c_audit_root)
    v7c_gate_root = _resolve_path(project_root, config.v7c_gate_root)

    v7b_audit = _read_json_object(v7b_audit_root / "failure_map_summary.json")
    v7c_audit = _read_json_object(v7c_audit_root / "failure_map_summary.json")
    v7c_gate = _read_json_object(v7c_gate_root / "v7_training_gate_summary.json")
    failure_priorities = _merged_failure_priorities(v7b_audit=v7b_audit, v7c_audit=v7c_audit)
    progress_bins = _merged_progress_bins(v7b_audit=v7b_audit, v7c_audit=v7c_audit)
    seed_rows = _collect_candidate_seed_rows(project_root=project_root, candidate_roots=config.candidate_roots)
    real_seed_review = _real_seed_review_summary(seed_rows)
    heldout_discovery = _heldout_discovery_summary(_resolve_path(project_root, config.dataset_search_root))
    two_stage_contract = _two_stage_contract()
    status = _preparation_status(real_seed_review)

    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "v4_fallback_policy": "preserved",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7d candidate metadata",
        "prompt_window_model": (
            "prompt-conditioned temporal sequence; preparation/rock-like ambiguity after the on-screen prompt "
            "is part of the bounded response window, not an independent final-label thumbnail"
        ),
        "heldout_discovery": heldout_discovery,
        "training_started": False,
        "dataset_generated": False,
        "seed_package_created": False,
        "promotion_eligible": False,
        "promotion_policy": _promotion_policy(),
        "v7b_audit_root": _display_path(v7b_audit_root, base=project_root),
        "v7c_audit_root": _display_path(v7c_audit_root, base=project_root),
        "v7c_gate_status": v7c_gate.get("status"),
        "v7c_gate_next_action": v7c_gate.get("next_action"),
        "failure_priorities": failure_priorities,
        "prompt_progress_bin_counts": progress_bins,
        "real_seed_review": real_seed_review,
        "two_stage_contract": two_stage_contract,
        "next_actions": _next_actions(real_seed_review),
        "claim_scope": (
            "status-only v7d preparation; does not run MediaPipe extraction, approve review rows, build seed NPZs, "
            "generate datasets, train models, validate MP4s, promote profiles, edit PDFs, or start final packaging"
        ),
    }

    _write_outputs(output_root=output_root, project_root=project_root, summary=summary, seed_rows=seed_rows)
    return summary


def _merged_failure_priorities(*, v7b_audit: Mapping[str, object], v7c_audit: Mapping[str, object]) -> list[dict[str, object]]:
    v7b_counts = _int_counter(_mapping(v7b_audit.get("group_counts")))
    v7c_counts = _int_counter(_mapping(v7c_audit.get("group_counts")))
    groups = sorted(set(v7b_counts) | set(v7c_counts))
    rows: list[dict[str, object]] = []
    for group in groups:
        combined = int(v7b_counts[group] + v7c_counts[group])
        if combined <= 0:
            continue
        rows.append(
            {
                "failure_group": group,
                "v7b_count": int(v7b_counts[group]),
                "v7c_count": int(v7c_counts[group]),
                "combined_count": combined,
                "v7d_response": _failure_response(group),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["combined_count"]), str(row["failure_group"])))


def _merged_progress_bins(*, v7b_audit: Mapping[str, object], v7c_audit: Mapping[str, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for audit in (v7b_audit, v7c_audit):
        counts.update(_int_counter(_mapping(audit.get("progress_bin_counts"))))
    return dict(sorted(counts.items()))


def _failure_response(group: str) -> str:
    responses = {
        "rock -> scissors": "stage1 rock/wait guard must suppress transition/scissors before sufficient prompt-window evidence",
        "paper -> scissors": "stage2 needs reviewed real paper recovery and late-geometry paper rescue evidence",
        "paper -> rock": "stage1 must avoid holding transition paper as rock/wait past the decision deadline",
        "scissors -> paper": "keep v7b delayed-scissors controls and do not add static early-scissors positives",
        "scissors -> rock": "stage1 must not over-abstain once scissors transition evidence appears",
    }
    return responses.get(group, "preserve diagnostics and add only reviewed non-heldout evidence before retraining")


def _collect_candidate_seed_rows(*, project_root: Path, candidate_roots: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate_root in candidate_roots:
        resolved_root = _resolve_path(project_root, candidate_root)
        if not resolved_root.exists():
            continue
        proposed_rows = _read_jsonl_if_exists(resolved_root / "proposed_segments.jsonl")
        review_rows = _read_review_rows(resolved_root / "segment_review_manifest.csv")
        review_by_id = {str(row.get("segment_id", "")).strip(): row for row in review_rows}
        for proposed in proposed_rows:
            row = dict(proposed)
            segment_id = str(row.get("segment_id", "")).strip()
            review = review_by_id.get(segment_id, {})
            for key in ("review_status", "approved_for_training", "review_notes", "quality_status"):
                if key in review:
                    row[key] = review[key]
            row["candidate_root"] = _display_path(resolved_root, base=project_root)
            _reject_heldout_metadata(row, context=resolved_root / "proposed_segments.jsonl")
            rows.append(_candidate_manifest_row(row))
    return rows


def _candidate_manifest_row(row: Mapping[str, object]) -> dict[str, object]:
    target = str(row.get("target_name", "")).strip().lower()
    role = str(row.get("proposal_role") or row.get("evidence_role") or "").strip()
    return {
        "segment_id": str(row.get("segment_id", "")).strip(),
        "target_name": target,
        "source_run_id": str(row.get("source_run_id", "")).strip(),
        "proposal_role": role,
        "v7d_seed_role": _v7d_seed_role(target_name=target, proposal_role=role),
        "quality_status": str(row.get("quality_status", "")).strip(),
        "review_status": str(row.get("review_status", "")).strip(),
        "approved_for_training": _truthy(row.get("approved_for_training")),
        "candidate_root": str(row.get("candidate_root", "")).strip(),
        "source_path": str(row.get("source_path", "")).strip(),
        "training_policy": str(row.get("training_policy") or "candidate_only_until_manual_segment_review_approval"),
    }


def _v7d_seed_role(*, target_name: str, proposal_role: str) -> str:
    text = f"{target_name} {proposal_role}".lower()
    if "paper" in text:
        return "hard_paper_prompt_window"
    if "rock_wait" in text or "wait" in text or "rock" in text:
        return "rock_wait_prompt_window"
    if "scissors" in text:
        return "scissors_boundary_control"
    return "unmapped_candidate"


def _real_seed_review_summary(seed_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    candidate_counts = Counter(str(row.get("target_name", "")) for row in seed_rows if str(row.get("target_name", "")).strip())
    approved_counts = Counter(
        str(row.get("target_name", ""))
        for row in seed_rows
        if _truthy(row.get("approved_for_training")) and str(row.get("review_status", "")).strip().lower() == "approved"
    )
    quality_pass_counts = Counter(
        str(row.get("target_name", ""))
        for row in seed_rows
        if str(row.get("quality_status", "")).strip().lower() == "auto_quality_pass"
    )
    roles_present = {role: any(str(row.get("v7d_seed_role", "")) == role for row in seed_rows) for role in REQUIRED_REAL_SEED_ROLES}
    approved_roles_present = {
        role: any(
            str(row.get("v7d_seed_role", "")) == role
            and _truthy(row.get("approved_for_training"))
            and str(row.get("review_status", "")).strip().lower() == "approved"
            for row in seed_rows
        )
        for role in REQUIRED_REAL_SEED_ROLES
    }
    blockers = []
    if not approved_roles_present["hard_paper_prompt_window"]:
        blockers.append("approved non-heldout hard paper prompt-window evidence is missing")
    if not approved_roles_present["rock_wait_prompt_window"]:
        blockers.append("approved non-heldout rock/wait prompt-window evidence is missing")
    if not roles_present["scissors_boundary_control"]:
        blockers.append("optional scissors boundary-control candidates are not yet present")
    return {
        "status": "ready_for_two_stage_dataset_design" if not blockers[:2] else "awaiting_reviewed_nonheldout_real_evidence",
        "candidate_segment_count": len(seed_rows),
        "candidate_counts_by_target": dict(sorted(candidate_counts.items())),
        "quality_pass_counts_by_target": dict(sorted(quality_pass_counts.items())),
        "approved_counts_by_target": dict(sorted(approved_counts.items())),
        "required_roles_present": roles_present,
        "approved_required_roles_present": approved_roles_present,
        "blockers": blockers,
        "review_policy": (
            "candidate rows are review aids only; v7d seed packages may use only manually approved non-heldout rows"
        ),
    }


def _preparation_status(real_seed_review: Mapping[str, object]) -> str:
    review_status = str(real_seed_review.get("status", ""))
    if review_status == "ready_for_two_stage_dataset_design":
        return "ready_for_v7d_two_stage_dataset_design"
    return "awaiting_v7d_real_seed_review"


def _two_stage_contract() -> dict[str, object]:
    return {
        "branch_label": BRANCH_LABEL,
        "stage1": {
            "name": "rock_wait_vs_transition_guard",
            "remap_mode": "rock_vs_transition",
            "label_names": ["rock", "transition"],
            "semantic_labels": {
                "rock": "rock_wait",
                "transition": "paper_or_scissors_response_transition",
            },
            "source_label_mapping": {
                "rock": "rock",
                "paper": "transition",
                "scissors": "transition",
            },
            "negative_pressure": "closed or ambiguous prompt-window sequences must not become high-confidence scissors",
        },
        "stage2": {
            "name": "paper_vs_scissors_resolver",
            "remap_mode": "paper_vs_scissors",
            "label_names": ["paper", "scissors"],
            "included_source_labels": ["paper", "scissors"],
            "excluded_source_labels": ["rock"],
            "paper_rescue": "mix fast paper recovery, delayed paper from early scissors-like ambiguity, and v4 late-geometry paper rescue",
            "scissors_policy": "preserve v7b delayed-scissors controls without adding static early-scissors positives",
        },
        "decision_policy": {
            "stage1_before_stage2": True,
            "response_window_deadline_progress": 0.50,
            "rock_wait_blocks_terminal_scissors": True,
            "stage2_used_only_after_transition_evidence": True,
            "fallback": "v4 remains the live/demo fallback until v7d passes strict MP4 and replay/live gates",
        },
        "existing_code_surfaces": {
            "dataset_remap_module": "embodied_rps.real_skeleton_dataset_remap",
            "stage1_remap_mode": "rock_vs_transition",
            "stage2_remap_mode": "paper_vs_scissors",
            "video_evaluator_module": "embodied_rps.tools.evaluate_two_stage_skeleton_video_predictions",
        },
    }


def _promotion_policy() -> dict[str, object]:
    return {
        "may_promote": False,
        "required_before_promotion": {
            "original20": "20/20 strict pass",
            "heldout15": "15/15 strict pass",
            "heldout_rock_false_triggers": 0,
            "paper_scissors_regression": "none",
            "replay_live": "diagnostics only after strict MP4 gates pass",
        },
        "fallback_if_failed": "keep v4 live/demo fallback and preserve v7d diagnostics",
    }


def _next_actions(real_seed_review: Mapping[str, object]) -> list[str]:
    actions = [
        "review or collect non-heldout paper prompt-window segments that resolve before the strict 0.50 decision deadline",
        "review or collect non-heldout rock/wait prompt-window hard negatives with closed/ambiguous preparation and no binary transition metadata",
        "use existing remap modes to build separate stage1 rock_vs_transition and stage2 paper_vs_scissors datasets only after reviewed evidence is ready",
        "train stage1/stage2 TCN seeds and validate in order: synthetic metrics, original20, heldout15, replay diagnostics, fresh live",
    ]
    blockers = real_seed_review.get("blockers")
    if isinstance(blockers, Sequence) and blockers:
        actions.insert(0, "do not start v7d training or promotion while reviewed real-seed blockers remain")
    return actions


def _heldout_discovery_summary(dataset_search_root: Path) -> dict[str, object]:
    if not dataset_search_root.exists():
        return {
            "status": "dataset_search_root_missing",
            "heldout_test_root_count": 0,
            "heldout_mp4_count": 0,
            "path_policy": "paths are intentionally omitted from the v7d preparation summary; heldout */test roots are validation-only",
        }
    test_roots = sorted(
        (path for path in dataset_search_root.rglob("test") if path.is_dir()),
        key=lambda path: path.as_posix(),
    )
    return {
        "status": "passed",
        "heldout_test_root_count": len(test_roots),
        "heldout_mp4_count": sum(len(list(root.rglob("*.mp4"))) for root in test_roots),
        "path_policy": "paths are intentionally omitted from the v7d preparation summary; heldout */test roots are validation-only",
    }


def _write_outputs(
    *,
    output_root: Path,
    project_root: Path,
    summary: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, object]],
) -> None:
    (output_root / "v7d_preparation_summary.json").write_text(
        json.dumps(_json_ready(dict(summary)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "v7d_two_stage_contract.json").write_text(
        json.dumps(_json_ready(summary["two_stage_contract"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_root / "v7d_seed_candidate_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in seed_rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True, ensure_ascii=False) + "\n")
    _write_failure_priority_csv(output_root / "v7d_failure_priority_map.csv", summary.get("failure_priorities", []))
    (output_root / "v7d_preparation_summary.md").write_text(_markdown(summary, project_root=project_root), encoding="utf-8")


def _write_failure_priority_csv(path: Path, rows: object) -> None:
    fieldnames = ["failure_group", "v7b_count", "v7c_count", "combined_count", "v7d_response"]
    row_values = rows if isinstance(rows, Sequence) else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_values:
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fieldnames})


def _markdown(summary: Mapping[str, object], *, project_root: Path) -> str:
    del project_root
    lines = [
        "# V7d Prompt-Window Guard Preparation",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Branch: `{summary.get('branch_label')}`",
        f"- v4 fallback: `{summary.get('v4_fallback_policy')}`",
        f"- Training started: `{summary.get('training_started')}`",
        f"- Dataset generated: `{summary.get('dataset_generated')}`",
        f"- Seed package created: `{summary.get('seed_package_created')}`",
        "- Heldout policy: validation-only `*/test` MP4s are rejected from candidate metadata.",
        "- Recording model: prompt-conditioned temporal response window.",
        "",
        "## Failure Priorities",
        "",
    ]
    priorities = summary.get("failure_priorities")
    if isinstance(priorities, Sequence):
        for row in priorities:
            if isinstance(row, Mapping):
                lines.append(
                    f"- `{row.get('failure_group')}`: combined={row.get('combined_count')} "
                    f"(v7b={row.get('v7b_count')}, v7c={row.get('v7c_count')})"
                )
    review = summary.get("real_seed_review")
    if isinstance(review, Mapping):
        lines.extend(
            [
                "",
                "## Real-Seed Review",
                "",
                f"- Status: `{review.get('status')}`",
                f"- Candidate segments: `{review.get('candidate_segment_count')}`",
                f"- Candidate counts: `{review.get('candidate_counts_by_target')}`",
                f"- Approved counts: `{review.get('approved_counts_by_target')}`",
                f"- Blockers: `{review.get('blockers')}`",
            ]
        )
    contract = summary.get("two_stage_contract")
    if isinstance(contract, Mapping):
        stage1 = _mapping(contract.get("stage1"))
        stage2 = _mapping(contract.get("stage2"))
        lines.extend(
            [
                "",
                "## Two-Stage Contract",
                "",
                f"- Stage1: `{stage1.get('remap_mode')}` labels `{stage1.get('label_names')}`",
                f"- Stage2: `{stage2.get('remap_mode')}` labels `{stage2.get('label_names')}`",
                "- Stage2 is used only after stage1 has enough transition evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
        ]
    )
    actions = summary.get("next_actions")
    if isinstance(actions, Sequence):
        for action in actions:
            lines.append(f"- {action}")
    lines.extend(["", "## Claim Scope", "", str(summary.get("claim_scope", "")), ""])
    return "\n".join(lines)


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required v7d preparation input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if not value:
            continue
        if _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test candidate path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _int_counter(value: Mapping[str, object]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for key, raw in value.items():
        try:
            counter[str(key)] += int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return counter


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = ["BRANCH_LABEL", "DEFAULT_OUTPUT_ROOT", "V7DPreparationConfig", "prepare_v7d_prompt_window_guard"]
