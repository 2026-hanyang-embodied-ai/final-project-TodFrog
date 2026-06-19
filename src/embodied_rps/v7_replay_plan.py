"""Replay planning artifacts for the v7 RPS validation branch."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


COUNTER_ACTIONS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


@dataclass(frozen=True)
class V7ReplayPlanConfig:
    """Configuration for v7 archived-live and approved-segment replay planning."""

    seed_package_root: Path = Path("artifacts/real_skeleton_v7_rps_seed_package_20260617")
    output_root: Path = Path("artifacts/real_skeleton_v7_replay_plan_20260617")
    profile_json_path: Path = Path("results/model_profiles/real_skeleton_three_class_wait_v7_rps_pose_tcn_ensemble.json")
    project_root: Path = Path.cwd()


def write_v7_replay_plan(config: V7ReplayPlanConfig) -> dict[str, object]:
    """Write replay manifests without running replay inference."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    manifest = _read_json(config.seed_package_root / "v7_seed_manifest.json")
    sources = manifest.get("sources", []) if isinstance(manifest, Mapping) else []
    source_items = [source for source in sources if isinstance(source, Mapping)]
    archived_rows = _archived_live_rows(source_items, project_root=config.project_root)
    proposals = _read_jsonl(config.seed_package_root / "proposed_segments.jsonl")
    approvals, review_failures = _read_review_manifest(config.seed_package_root / "segment_review_manifest.csv")
    failures.extend(review_failures)
    segment_rows, segment_failures = _approved_segment_rows(
        proposals=proposals,
        approvals=approvals,
        seed_package_root=config.seed_package_root,
        project_root=config.project_root,
    )
    failures.extend(segment_failures)

    profile = _profile_status(config.profile_json_path)
    archived_path = config.output_root / "archived_live_replay_manifest.jsonl"
    segment_path = config.output_root / "approved_segment_replay_manifest.jsonl"
    _write_jsonl(archived_path, archived_rows)
    _write_jsonl(segment_path, segment_rows)

    status = _status(failures=failures, approved_segment_count=len(segment_rows), profile_status=str(profile["status"]))
    summary = {
        "status": status,
        "seed_package_root": config.seed_package_root.as_posix(),
        "output_root": config.output_root.as_posix(),
        "profile": profile,
        "archived_live_replay": {
            "status": "ready_for_profile_replay" if archived_rows else "missing_archived_live_sources",
            "entry_count": len(archived_rows),
            "manifest_path": archived_path.as_posix(),
        },
        "approved_segment_replay": {
            "status": "ready_for_profile_replay" if segment_rows else "awaiting_manual_segment_approval",
            "entry_count": len(segment_rows),
            "manifest_path": segment_path.as_posix(),
        },
        "failures": failures,
        "commands": _commands(config=config, archived_manifest=archived_path, segment_manifest=segment_path),
        "notes": [
            "This plan does not run model inference or promote v7.",
            "Archived live runs used as training evidence are replay diagnostics, not final validation evidence.",
            "Held-out test paths are rejected from approved segment replay manifests.",
        ],
    }
    _write_summary(config.output_root, summary)
    return summary


def _archived_live_rows(sources: Sequence[Mapping[str, object]], *, project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        if source.get("kind") != "archived_live_run":
            continue
        gesture = str(source.get("expected_actual_gesture", "")).lower()
        expected_action = COUNTER_ACTIONS.get(gesture)
        rows.append(
            {
                "run_id": source.get("source_id"),
                "source_path": _display_path(source.get("path"), project_root=project_root),
                "overlay_video": _display_path(source.get("overlay_video"), project_root=project_root),
                "expected_actual_gesture": gesture,
                "expected_robot_action": expected_action,
                "prior_passed": source.get("passed"),
                "evidence_role": source.get("evidence_role"),
                "replay_policy": source.get("replay_policy"),
                "profile_required": True,
            }
        )
    return rows


def _approved_segment_rows(
    *,
    proposals: Sequence[Mapping[str, object]],
    approvals: Mapping[str, Mapping[str, str]],
    seed_package_root: Path,
    project_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    proposal_by_id = {str(proposal.get("segment_id")): proposal for proposal in proposals}
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for segment_id, approval in approvals.items():
        if not _approved(approval):
            continue
        proposal = proposal_by_id.get(segment_id)
        if proposal is None:
            failures.append({"code": "approved_segment_missing_from_proposals", "segment_id": segment_id})
            continue
        source_path = str(proposal.get("source_path", ""))
        if _contains_heldout_test_component(source_path):
            failures.append({"code": "heldout_segment_approved_for_replay", "segment_id": segment_id, "source_path": source_path})
            continue
        if proposal.get("quality_status") != "auto_quality_pass":
            failures.append({"code": "approved_segment_failed_auto_quality", "segment_id": segment_id})
            continue
        target = str(proposal.get("target_name", "")).lower()
        rows.append(
            {
                "segment_id": segment_id,
                "target_name": target,
                "expected_robot_action": COUNTER_ACTIONS.get(target),
                "skeleton_npz": _display_path(proposal.get("skeleton_npz"), project_root=project_root, fallback_root=seed_package_root),
                "source_run_id": proposal.get("source_run_id"),
                "source_name": proposal.get("source_name"),
                "proposal_role": proposal.get("proposal_role"),
                "profile_required": True,
            }
        )
    return rows, failures


def _approved(row: Mapping[str, str]) -> bool:
    return row.get("approved_for_training", "").strip().lower() == "true" and row.get("review_status", "").strip().lower() == "approved"


def _contains_heldout_test_component(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.endswith("/test")


def _profile_status(profile_json_path: Path) -> dict[str, object]:
    profile_pt_path = profile_json_path.with_suffix(".pt")
    if not profile_json_path.exists() or not profile_pt_path.exists():
        return {
            "status": "missing",
            "profile_json": profile_json_path.as_posix(),
            "profile_pt": profile_pt_path.as_posix(),
            "json_exists": profile_json_path.exists(),
            "pt_exists": profile_pt_path.exists(),
        }
    try:
        profile = json.loads(profile_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "profile_json": profile_json_path.as_posix(), "failure": str(exc)}
    model = profile.get("model")
    labels = profile.get("label_names")
    if model != "tcn":
        return {"status": "wrong_model", "profile_json": profile_json_path.as_posix(), "model": model, "label_names": labels}
    if labels != ["rock", "paper", "scissors"]:
        return {"status": "invalid_labels", "profile_json": profile_json_path.as_posix(), "model": model, "label_names": labels}
    return {"status": "passed", "profile_json": profile_json_path.as_posix(), "profile_pt": profile_pt_path.as_posix(), "model": model, "label_names": labels}


def _status(*, failures: Sequence[Mapping[str, object]], approved_segment_count: int, profile_status: str) -> str:
    if failures:
        return "invalid_replay_plan"
    if approved_segment_count == 0:
        return "awaiting_manual_segment_approval"
    if profile_status != "passed":
        return "awaiting_v7_profile"
    return "ready_for_replay_execution"


def _commands(*, config: V7ReplayPlanConfig, archived_manifest: Path, segment_manifest: Path) -> dict[str, object]:
    return {
        "archived_live_replay_manifest": archived_manifest.as_posix(),
        "approved_segment_replay_manifest": segment_manifest.as_posix(),
        "profile_json": config.profile_json_path.as_posix(),
        "replay_status_note": "Run replay inference only after the v7 TCN profile exists.",
    }


def _display_path(value: object, *, project_root: Path, fallback_root: Path | None = None) -> str | None:
    if value is None:
        return None
    path_text = str(value)
    path = Path(path_text)
    bases = [project_root]
    if fallback_root is not None:
        bases.insert(0, fallback_root)
    for base in bases:
        try:
            return path.resolve(strict=False).relative_to(base.resolve(strict=False)).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _read_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(value)
    return rows


def _read_review_manifest(path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
    if not path.exists():
        return {}, []
    approvals: dict[str, dict[str, str]] = {}
    duplicate_ids: set[str] = set()
    failures: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            segment_id = str(row.get("segment_id", "")).strip()
            if not segment_id:
                continue
            if segment_id in approvals or segment_id in duplicate_ids:
                duplicate_ids.add(segment_id)
                approvals.pop(segment_id, None)
                failures.append(
                    {
                        "code": "segment_review_manifest_duplicate_segment_ids",
                        "segment_id": segment_id,
                        "row_number": row_number,
                    }
                )
                continue
            approvals[segment_id] = dict(row)
    return approvals, failures


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    (output_root / "v7_replay_plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "v7_replay_plan_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    archived = summary.get("archived_live_replay")
    segment = summary.get("approved_segment_replay")
    profile = summary.get("profile")
    archived_map = archived if isinstance(archived, Mapping) else {}
    segment_map = segment if isinstance(segment, Mapping) else {}
    profile_map = profile if isinstance(profile, Mapping) else {}
    lines = [
        "# V7 Replay Plan Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- archived live entries: `{archived_map.get('entry_count')}`",
        f"- approved segment entries: `{segment_map.get('entry_count')}`",
        f"- profile status: `{profile_map.get('status')}`",
        "",
        "## Manifests",
        "",
        f"- archived live replay: `{archived_map.get('manifest_path')}`",
        f"- approved segment replay: `{segment_map.get('manifest_path')}`",
        "",
        "## Notes",
        "",
    ]
    notes = summary.get("notes")
    if isinstance(notes, list):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["V7ReplayPlanConfig", "write_v7_replay_plan"]
