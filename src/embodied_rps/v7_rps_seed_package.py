"""V7 RPS seed inventory, segment review, and approved seed packaging."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.real_skeleton_training import landmark_velocity_features
from embodied_rps.three_class_wait_skeletons import TARGET_TO_LABEL

TARGET_NAMES: tuple[str, ...] = ("rock", "paper", "scissors")
V7_SEED_NPZ_REQUIRED_KEYS: tuple[str, ...] = (
    "sample_ids",
    "labels",
    "label_names",
    "target_names",
    "split_names",
    "lengths",
    "mask",
    "progress",
    "canonical_landmarks",
    "features",
    "source_names",
    "hard_example_flags",
    "source_paths",
    "proposal_roles",
    "source_run_ids",
    "source_metadata_json",
)
DEFAULT_V4_SEED_ROOT = Path("artifacts/real_skeleton_v4_calibration_seed_package_fewshot_20260615")
DEFAULT_LIVE_ROCK_SEED_ROOT = Path("artifacts/live_rock_false_trigger_overlay_seed_20260616")
DEFAULT_ARCHIVE_ROOT = Path("artifacts/realtime_demo_run_archive_20260616")
DEFAULT_SCISSORS_COLLECTION_ROOT = Path("artifacts/realtime_scissors_pose_collection_20260617")
DEFAULT_V7_OUTPUT_ROOT = Path("artifacts/real_skeleton_v7_rps_seed_package_20260617")
ARCHIVED_LIVE_EVIDENCE_ROLES: dict[str, str] = {
    "run_20260617_122033": "rock_false_trigger_hard_negative",
    "run_20260617_141126": "varied_rock_positive_and_wait_guard",
    "run_20260617_141740": "hard_paper_vs_scissors_confusion",
    "run_20260617_142926": "short_successful_scissors_retake",
}


@dataclass(frozen=True)
class V7SeedManifestConfig:
    """Inputs for writing the v7 source inventory manifest."""

    project_root: Path
    output_root: Path = DEFAULT_V7_OUTPUT_ROOT
    dataset_search_root: Path = Path("D:/dataset")
    v4_seed_root: Path = DEFAULT_V4_SEED_ROOT
    live_rock_seed_root: Path = DEFAULT_LIVE_ROCK_SEED_ROOT
    archive_root: Path = DEFAULT_ARCHIVE_ROOT
    scissors_collection_root: Path = DEFAULT_SCISSORS_COLLECTION_ROOT


@dataclass(frozen=True)
class V7SegmentProposalConfig:
    """Inputs for proposing review-gated v7 real skeleton segments."""

    run_roots: tuple[Path, ...]
    output_root: Path = DEFAULT_V7_OUTPUT_ROOT
    transition_run_ids: tuple[str, ...] = ("run_20260617_150616",)
    static_run_ids: tuple[str, ...] = ("run_20260617_150023",)
    sequence_length: int = 72
    prefix_frames: int = 24
    static_stride: int = 144
    min_segment_frames: int = 30
    min_detection_coverage: float = 0.95


@dataclass(frozen=True)
class V7ArchivedLiveOverlayProposalConfig:
    """Inputs for proposing review-gated archived live overlay skeleton candidates."""

    project_root: Path
    output_root: Path = DEFAULT_V7_OUTPUT_ROOT
    archive_root: Path = DEFAULT_ARCHIVE_ROOT
    run_ids: tuple[str, ...] = tuple(ARCHIVED_LIVE_EVIDENCE_ROLES)
    sequence_length: int = 72
    prefix_frames: int = 24
    min_segment_frames: int = 30
    min_detection_coverage: float = 0.95
    extract_missing_sidecars: bool = False
    overwrite_extracted_sidecars: bool = False


def write_v7_seed_manifest(config: V7SeedManifestConfig) -> dict[str, object]:
    """Write a machine-readable inventory of v7 source candidates and roles."""

    output_root = _under_project(config.project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, object]] = []

    v4_root = _under_project(config.project_root, config.v4_seed_root)
    v4_summary = _read_json_if_exists(v4_root / "seed_package_summary.json")
    v4_status = str(v4_summary.get("status", "missing")) if v4_summary else "missing"
    sources.append(
        {
            "source_id": "v4_calibration_fewshot",
            "kind": "seed_package",
            "path": _path_from_base(v4_root, base=config.project_root),
            "status": v4_status,
            "role": "approved_training_seed" if v4_status == "passed" else "unavailable",
            "target_counts": v4_summary.get("target_counts", {}) if v4_summary else {},
            "policy": "accepted non-held-out 5/5/5 few-shot calibration seeds",
        }
    )

    live_root = _under_project(config.project_root, config.live_rock_seed_root)
    live_summary = _read_json_if_exists(live_root / "seed_package_summary.json")
    live_status = str(live_summary.get("status", "missing")) if live_summary else "missing"
    live_training_use = str(live_summary.get("training_use", "")) if live_summary else ""
    sources.append(
        {
            "source_id": "live_rock_false_trigger_overlay",
            "kind": "seed_package",
            "path": _path_from_base(live_root, base=config.project_root),
            "status": live_status,
            "role": "approved_training_seed"
            if live_status == "passed" and live_training_use == "accepted_overlay_derived_hard_negative"
            else "unavailable",
            "target_counts": {"rock": int(live_summary.get("segment_count", 0))} if live_summary else {},
            "policy": "accepted overlay-derived rock hard-negative evidence",
        }
    )

    archive_root = _under_project(config.project_root, config.archive_root)
    for run_id, evidence_role in ARCHIVED_LIVE_EVIDENCE_ROLES.items():
        run_root = archive_root / run_id
        postcapture = _read_json_if_exists(run_root / "reports" / "postcapture_summary.json")
        media_path = run_root / "media" / "live_camera_overlay.mp4"
        gate = _mapping(postcapture.get("demo_success_gate", {}) if postcapture else {})
        sources.append(
            {
                "source_id": run_id,
                "kind": "archived_live_run",
                "path": _path_from_base(run_root, base=config.project_root),
                "status": "present" if run_root.exists() else "missing",
                "role": "candidate_after_overlay_extraction" if run_root.exists() else "unavailable",
                "evidence_role": evidence_role,
                "expected_actual_gesture": gate.get("expected_actual_gesture"),
                "passed": gate.get("passed"),
                "overlay_video": _path_from_base(media_path, base=config.project_root) if media_path.exists() else None,
                "replay_policy": "not final validation evidence after used as a training seed",
            }
        )

    collection_root = _under_project(config.project_root, config.scissors_collection_root)
    for run_id, evidence_role in {
        "run_20260617_150023": "scissors_static_and_viewpoint_seed_candidate",
        "run_20260617_150616": "preferred_rock_standby_to_scissors_transition_seed_candidate",
    }.items():
        run_root = collection_root / run_id
        quality = _read_json_if_exists(run_root / "summary" / "quality_summary.json")
        sources.append(
            {
                "source_id": run_id,
                "kind": "scissors_pose_collection",
                "path": _path_from_base(run_root, base=config.project_root),
                "status": str(quality.get("status", "present")) if quality else ("present" if run_root.exists() else "missing"),
                "role": "candidate_after_segment_review" if run_root.exists() else "unavailable",
                "evidence_role": evidence_role,
                "frame_count": quality.get("frame_count") if quality else None,
                "detection_rate": quality.get("detection_rate") if quality else None,
            }
        )

    heldout_roots = _discover_heldout_test_roots(config.dataset_search_root)
    heldout_count = sum(len(list(root.rglob("*.mp4"))) for root in heldout_roots)
    sources.append(
        {
            "source_id": "heldout15",
            "kind": "validation_mp4_root",
            "path": [root.as_posix() for root in heldout_roots],
            "status": "present" if heldout_count else "missing",
            "role": "validation_only",
            "mp4_count": heldout_count,
            "policy": "excluded from all seed packages and training datasets",
        }
    )

    manifest = {
        "status": "passed",
        "branch": "v7_rps_pose",
        "project_root": ".",
        "output_root": _path_from_base(output_root, base=config.project_root),
        "sources": sources,
        "heldout_policy": "held-out test MP4s are validation-only and must not enter v7 seeds",
        "review_gate": "segment_review_manifest.csv approvals required before seed NPZ generation",
    }
    manifest_path = output_root / "v7_seed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "passed",
        "manifest": _path_from_base(manifest_path, base=config.project_root),
        "source_count": len(sources),
        "heldout_mp4_count": heldout_count,
    }
    return summary


def write_v7_archived_live_candidate_manifest(
    *,
    project_root: Path,
    output_root: Path = DEFAULT_V7_OUTPUT_ROOT,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> dict[str, object]:
    """Write review-safe archived live seed-candidate status without extracting seeds."""

    resolved_output_root = _under_project(project_root, output_root)
    resolved_archive_root = _under_project(project_root, archive_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for run_id, evidence_role in ARCHIVED_LIVE_EVIDENCE_ROLES.items():
        rows.append(
            _archived_live_candidate_row(
                project_root=project_root,
                archive_root=resolved_archive_root,
                run_id=run_id,
                evidence_role=evidence_role,
            )
        )

    manifest_path = resolved_output_root / "archived_live_seed_candidate_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")

    status_counts = Counter(str(row.get("candidate_status", "")) for row in rows)
    summary = {
        "status": "passed",
        "output_root": _path_from_base(resolved_output_root, base=project_root),
        "manifest": _path_from_base(manifest_path, base=project_root),
        "entry_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "review_policy": "archived live runs are diagnostic/extraction-pending until overlay MediaPipe extraction passes and manual review approves explicit segment rows",
        "seed_review_manifest_unchanged": _path_from_base(resolved_output_root / "segment_review_manifest.csv", base=project_root),
        "claim_scope": "status-only archived live candidate manifest; does not run MediaPipe, create skeleton NPZ sidecars, approve rows, build seeds, train, validate, or promote",
    }
    summary_json = resolved_output_root / "archived_live_seed_candidate_summary.json"
    summary_md = resolved_output_root / "archived_live_seed_candidate_summary.md"
    summary_json.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_archived_live_candidate_summary_md(summary_md, summary=summary, rows=rows, output_root=resolved_output_root)
    return summary


def _archived_live_candidate_row(
    *,
    project_root: Path,
    archive_root: Path,
    run_id: str,
    evidence_role: str,
) -> dict[str, object]:
    run_root = archive_root / run_id
    overlay_video = run_root / "media" / "live_camera_overlay.mp4"
    frame_log = run_root / "media" / "live_camera_frames.jsonl"
    postcapture = run_root / "reports" / "postcapture_summary.json"
    skeleton_sidecars = sorted(run_root.rglob("*skeleton*.npz")) if run_root.exists() else []
    rows = _read_jsonl_if_exists(frame_log)
    gate = _mapping(_read_json_if_exists(postcapture).get("demo_success_gate", {}))
    expected_actual = str(gate.get("expected_actual_gesture") or _most_common(row.get("expected_actual_gesture") for row in rows) or "")
    detected_count = sum(1 for row in rows if bool(row.get("detected")))
    response_count = sum(1 for row in rows if bool(row.get("response_window")))
    has_landmark_payload = any(
        any(key in row for key in ("landmarks", "canonical_landmarks", "hand_landmarks", "world_landmarks"))
        for row in rows[:10]
    )
    if not run_root.exists():
        candidate_status = "missing_archive_run"
        next_action = "restore archive run before considering overlay extraction"
    elif not overlay_video.exists() or not frame_log.exists():
        candidate_status = "missing_required_archive_artifacts"
        next_action = "restore overlay video and frame log before considering overlay extraction"
    elif skeleton_sidecars:
        candidate_status = "skeleton_sidecar_present_manual_review_required"
        next_action = "quality-check sidecar and add explicit review rows only if non-held-out and visually approved"
    else:
        candidate_status = "overlay_extraction_pending"
        next_action = "run MediaPipe overlay extraction before any seed review row can be considered"
    return {
        "run_id": run_id,
        "kind": "archived_live_run",
        "archive_root": _path_from_base(run_root, base=project_root),
        "evidence_role": evidence_role,
        "target_name": expected_actual,
        "candidate_status": candidate_status,
        "training_policy": "diagnostic_only_until_overlay_extraction_quality_check_and_manual_approval",
        "review_manifest_policy": "not_present_in_segment_review_manifest",
        "overlay_video": _path_from_base(overlay_video, base=project_root) if overlay_video.exists() else None,
        "overlay_video_exists": overlay_video.exists(),
        "overlay_video_bytes": overlay_video.stat().st_size if overlay_video.exists() else 0,
        "frame_log": _path_from_base(frame_log, base=project_root) if frame_log.exists() else None,
        "frame_log_exists": frame_log.exists(),
        "frame_log_record_count": len(rows),
        "detected_frame_count": detected_count,
        "detection_coverage": detected_count / max(1, len(rows)),
        "response_window_frame_count": response_count,
        "has_landmark_payload_in_frame_log": has_landmark_payload,
        "skeleton_sidecar_count": len(skeleton_sidecars),
        "skeleton_sidecars": [_path_from_base(path, base=project_root) for path in skeleton_sidecars],
        "demo_success_passed": gate.get("passed"),
        "expected_actual_gesture": expected_actual,
        "first_response_prompt_binary_decision": gate.get("first_response_prompt_binary_decision"),
        "next_action": next_action,
    }


def _write_archived_live_candidate_summary_md(
    path: Path,
    *,
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    output_root: Path,
) -> Path:
    lines = [
        "# V7 Archived Live Seed Candidate Status",
        "",
        "This is a status-only review aid. It does not extract MediaPipe landmarks, add segment-review rows, approve seeds, train, validate, or promote v7.",
        "",
        "## Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- entries: `{summary.get('entry_count')}`",
        f"- status counts: `{summary.get('status_counts')}`",
        f"- review policy: `{summary.get('review_policy')}`",
        "",
        "## Archived Runs",
        "",
        "| Run | Target | Role | Candidate Status | Frames | Detection | Sidecars | Next Action |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_id", "")),
                    str(row.get("target_name", "")),
                    str(row.get("evidence_role", "")),
                    str(row.get("candidate_status", "")),
                    str(row.get("frame_log_record_count", "")),
                    f"{float(row.get('detection_coverage', 0.0)):.4f}",
                    str(row.get("skeleton_sidecar_count", "")),
                    str(row.get("next_action", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def propose_v7_archived_live_overlay_segments(config: V7ArchivedLiveOverlayProposalConfig) -> dict[str, object]:
    """Propose quality-screened archived live skeleton segments for manual review only."""

    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if config.prefix_frames < 0:
        raise ValueError("prefix_frames must be non-negative")
    if config.min_segment_frames <= 0:
        raise ValueError("min_segment_frames must be positive")
    output_root = _under_project(config.project_root, config.output_root)
    archive_root = _under_project(config.project_root, config.archive_root)
    output_root.mkdir(parents=True, exist_ok=True)
    segments_root = output_root / "segments"
    previews_root = output_root / "previews"
    segments_root.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)

    archive_records: list[dict[str, object]] = []
    skipped_runs: list[dict[str, object]] = []
    extraction_summaries: list[dict[str, object]] = []
    for run_id in config.run_ids:
        run_root = archive_root / run_id
        evidence_role = ARCHIVED_LIVE_EVIDENCE_ROLES.get(run_id, "archived_live_overlay_candidate")
        sidecar_path = _select_archived_live_skeleton_sidecar(run_root)
        if sidecar_path is None and config.extract_missing_sidecars:
            extraction = _extract_archived_live_overlay_sidecar(
                run_root=run_root,
                output_path=run_root / "media" / "live_camera_skeletons.npz",
                overwrite=config.overwrite_extracted_sidecars,
            )
            extraction_summaries.append(_relative_extraction_summary(extraction, project_root=config.project_root))
            if extraction.get("status") == "passed":
                sidecar_path = Path(str(extraction["skeleton_npz"]))
        if sidecar_path is None:
            skipped_runs.append(
                {
                    "run_id": run_id,
                    "status": "missing_skeleton_sidecar",
                    "next_action": "run with extract_missing_sidecars after installing mediapipe, then rerun proposals",
                }
            )
            continue
        frame_log = run_root / "media" / "live_camera_frames.jsonl"
        postcapture = run_root / "reports" / "postcapture_summary.json"
        frame_rows = _read_jsonl_if_exists(frame_log)
        gate = _mapping(_read_json_if_exists(postcapture).get("demo_success_gate", {}))
        target_name = str(gate.get("expected_actual_gesture") or _most_common(row.get("expected_actual_gesture") for row in frame_rows) or "")
        if target_name not in TARGET_NAMES:
            skipped_runs.append({"run_id": run_id, "status": "invalid_target_name", "target_name": target_name})
            continue
        skeleton = _load_collection_skeleton_npz(sidecar_path)
        try:
            archive_records.append(
                _write_archived_live_segment_record(
                    run_id=run_id,
                    evidence_role=evidence_role,
                    target_name=target_name,
                    run_root=run_root,
                    frame_rows=frame_rows,
                    skeleton=skeleton,
                    sidecar_path=sidecar_path,
                    sequence_length=config.sequence_length,
                    prefix_frames=config.prefix_frames,
                    min_segment_frames=config.min_segment_frames,
                    min_detection_coverage=config.min_detection_coverage,
                    segments_root=segments_root,
                    previews_root=previews_root,
                    output_root=output_root,
                )
            )
        except ValueError as exc:
            skipped_runs.append({"run_id": run_id, "status": "segment_proposal_failed", "reason": str(exc)})

    archived_path = output_root / "archived_live_overlay_proposed_segments.jsonl"
    with archived_path.open("w", encoding="utf-8") as handle:
        for record in archive_records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")

    proposed_path = output_root / "proposed_segments.jsonl"
    existing_records = _read_jsonl_if_exists(proposed_path)
    if archive_records or not proposed_path.exists():
        merged_records = _merge_archived_live_proposals(existing_records, archive_records, run_ids=set(config.run_ids))
        review_decisions = _read_review_manifest_decisions(output_root / "segment_review_manifest.csv")
        with proposed_path.open("w", encoding="utf-8") as handle:
            for record in merged_records:
                handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
        review_manifest = _write_review_manifest_preserving_decisions(
            output_root / "segment_review_manifest.csv",
            merged_records,
            review_decisions,
        )
        pass_csv = _write_auto_quality_pass_csv(output_root / "auto_quality_pass_segments.csv", merged_records)
        packet_md = _write_segment_review_packet(output_root / "segment_review_packet.md", merged_records)
        contact_sheet = _write_review_contact_sheet(output_root / "review_contact_sheet.png", merged_records)
    else:
        merged_records = existing_records
        review_manifest = output_root / "segment_review_manifest.csv"
        pass_csv = output_root / "auto_quality_pass_segments.csv"
        packet_md = output_root / "segment_review_packet.md"
        contact_sheet = output_root / "review_contact_sheet.png"

    quality_counts = Counter(str(record.get("quality_status", "")) for record in archive_records)
    all_quality_counts = Counter(str(record.get("quality_status", "")) for record in merged_records)
    review_rows = list(csv.DictReader(review_manifest.open(encoding="utf-8"))) if review_manifest.exists() else []
    approved_count = sum(
        1
        for row in review_rows
        if _truthy(row.get("approved_for_training", "")) and str(row.get("review_status", "")).strip().lower() == "approved"
    )
    summary = {
        "status": "awaiting_manual_review",
        "output_root": _path_from_base(output_root, base=config.project_root),
        "proposed_segments": _path_from_base(proposed_path, base=config.project_root),
        "archived_live_overlay_proposed_segments": _path_from_base(archived_path, base=config.project_root),
        "review_manifest": _path_from_base(review_manifest, base=config.project_root),
        "auto_quality_pass_segments": _path_from_base(pass_csv, base=config.project_root),
        "segment_review_packet": _path_from_base(packet_md, base=config.project_root),
        "review_contact_sheet": _path_from_base(contact_sheet, base=config.project_root),
        "proposed_segment_count": len(archive_records),
        "merged_proposed_segment_count": len(merged_records),
        "auto_quality_pass_count": int(quality_counts.get("auto_quality_pass", 0)),
        "auto_quality_failed_count": int(quality_counts.get("auto_quality_failed", 0)),
        "merged_auto_quality_pass_count": int(all_quality_counts.get("auto_quality_pass", 0)),
        "approved_segment_count": approved_count,
        "skipped_runs": skipped_runs,
        "extraction_summaries": extraction_summaries,
        "review_policy": "archived live overlay candidates are pending manual review and are never auto-approved",
        "seed_package_created": False,
        "next_action": "review archived live overlay candidates visually before approving any rows",
    }
    summary_json = output_root / "archived_live_overlay_segment_summary.json"
    summary_md = output_root / "archived_live_overlay_segment_summary.md"
    summary_json.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_archived_live_overlay_segment_summary_md(summary_md, summary=summary, records=archive_records)
    return summary


def propose_v7_rps_segments(config: V7SegmentProposalConfig) -> dict[str, object]:
    """Propose quality-screened v7 scissors collection segments for manual review."""

    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if config.prefix_frames < 0:
        raise ValueError("prefix_frames must be non-negative")
    config.output_root.mkdir(parents=True, exist_ok=True)
    segments_root = config.output_root / "segments"
    previews_root = config.output_root / "previews"
    segments_root.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for run_root in config.run_roots:
        run_records = _read_jsonl(run_root / "scissors_pose_collection_frames.jsonl")
        skeleton = _load_collection_skeleton_npz(run_root / "scissors_pose_collection_skeletons.npz")
        if len(run_records) != skeleton["frame_count"]:
            raise ValueError(f"{run_root} frame log and skeleton NPZ length mismatch")
        run_id = run_root.name
        if run_id in set(config.static_run_ids):
            records.extend(
                _static_scissors_segments(
                    run_id=run_id,
                    run_root=run_root,
                    skeleton=skeleton,
                    config=config,
                    segments_root=segments_root,
                    previews_root=previews_root,
                )
            )
        if run_id in set(config.transition_run_ids):
            records.extend(
                _transition_scissors_segments(
                    run_id=run_id,
                    run_root=run_root,
                    skeleton=skeleton,
                    config=config,
                    segments_root=segments_root,
                    previews_root=previews_root,
                )
            )
        if run_id not in set(config.static_run_ids) and run_id not in set(config.transition_run_ids):
            prompt_names = set(cast(list[str], skeleton["active_prompts"]))
            if prompt_names == {"scissors"}:
                records.extend(
                    _static_scissors_segments(
                        run_id=run_id,
                        run_root=run_root,
                        skeleton=skeleton,
                        config=config,
                        segments_root=segments_root,
                        previews_root=previews_root,
                    )
                )
            elif "scissors" in prompt_names:
                records.extend(
                    _transition_scissors_segments(
                        run_id=run_id,
                        run_root=run_root,
                        skeleton=skeleton,
                        config=config,
                        segments_root=segments_root,
                        previews_root=previews_root,
                    )
                )

    proposed_path = config.output_root / "proposed_segments.jsonl"
    with proposed_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
    review_manifest = _write_review_manifest(config.output_root / "segment_review_manifest.csv", records)
    pass_csv = _write_auto_quality_pass_csv(config.output_root / "auto_quality_pass_segments.csv", records)
    review_packet = _write_segment_review_packet(config.output_root / "segment_review_packet.md", records)
    contact_sheet = _write_review_contact_sheet(config.output_root / "review_contact_sheet.png", records)
    summary_base = Path.cwd()
    summary = {
        "status": "awaiting_manual_review" if records else "no_segments_proposed",
        "output_root": _display_path(config.output_root, base=summary_base),
        "proposed_segment_count": len(records),
        "quality_pass_count": sum(1 for record in records if record.get("quality_status") == "auto_quality_pass"),
        "proposed_segments": _display_path(proposed_path, base=summary_base),
        "segment_review_manifest": _display_path(review_manifest, base=summary_base),
        "auto_quality_pass_segments": _display_path(pass_csv, base=summary_base),
        "segment_review_packet": _display_path(review_packet, base=summary_base),
        "review_contact_sheet": _display_path(contact_sheet, base=summary_base),
        "review_gate": "manual approval required before v7_rps_seed_dataset.npz is written",
    }
    (config.output_root / "segment_proposal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_v7_rps_seed_package(*, output_root: Path, sequence_length: int = 72) -> dict[str, object]:
    """Build the v7 seed NPZ only from explicitly approved review rows."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")
    readiness = audit_v7_segment_review_readiness(output_root=output_root)
    readiness_status = str(readiness.get("status", ""))
    if readiness_status not in {"awaiting_manual_segment_approval", "ready_for_seed_package_build"}:
        readiness_failures = cast(list[dict[str, object]], readiness.get("failures", []))
        readiness_failure_codes = {str(failure.get("code", "")) for failure in readiness_failures}
        if any(code.startswith("review_contact_sheet") or code.startswith("review_preview") for code in readiness_failure_codes):
            raise ValueError(f"v7 review visual artifacts failed: {readiness_failures}")
        raise ValueError(f"v7 review readiness failed before seed packaging: {readiness_failures}")
    records = {str(record["segment_id"]): record for record in _read_jsonl(proposed_path)}
    with review_path.open(encoding="utf-8") as handle:
        review_reader = csv.DictReader(handle)
        review_fieldnames = list(review_reader.fieldnames or [])
        review_rows = list(review_reader)
    approved_rows = [
        row
        for row in review_rows
        if _truthy(row.get("approved_for_training", "")) and str(row.get("review_status", "")).lower() == "approved"
    ]
    if not approved_rows:
        summary = {
            "status": "awaiting_manual_segment_approval",
            "output_root": output_root.as_posix(),
            "approved_segment_count": 0,
            "review_manifest": review_path.as_posix(),
            "seed_npz": (output_root / "v7_rps_seed_dataset.npz").as_posix(),
        }
        return summary

    visual_audit = _audit_review_visual_artifacts(output_root=output_root, records=records.values())
    if visual_audit["status"] == "failed":
        raise ValueError(f"v7 review visual artifacts failed: {visual_audit['failures']}")

    seed_records: list[dict[str, object]] = []
    for row in approved_rows:
        segment_id = str(row["segment_id"])
        if segment_id not in records:
            raise ValueError(f"approved segment {segment_id} is missing from proposed_segments.jsonl")
        record = records[segment_id]
        if record.get("quality_status") != "auto_quality_pass":
            raise ValueError(f"approved segment {segment_id} did not pass automated quality checks")
        source_path = str(record.get("source_path", ""))
        _reject_heldout_path(source_path, proposed_path)
        seed_records.append(
            _seed_record_from_segment(
                record,
                review_row=row,
                sequence_length=sequence_length,
                base=output_root,
            )
        )

    validation = _validate_seed_records(seed_records, sequence_length=sequence_length)
    if validation["status"] != "passed":
        raise ValueError(f"v7 seed package validation failed: {validation['failures']}")
    npz_path = output_root / "v7_rps_seed_dataset.npz"
    metadata_path = output_root / "seed_metadata.jsonl"
    quality_path = output_root / "seed_quality_summary.csv"
    _write_seed_npz(npz_path, seed_records, sequence_length=sequence_length)
    _write_seed_metadata(metadata_path, seed_records)
    _write_seed_quality_csv(quality_path, seed_records)
    target_counts = dict(sorted(Counter(str(record["target_name"]) for record in seed_records).items()))
    summary = {
        "status": "passed",
        "review_gate": "manual_approved",
        "output_root": output_root.as_posix(),
        "seed_npz": npz_path.as_posix(),
        "seed_metadata": metadata_path.as_posix(),
        "seed_quality_summary": quality_path.as_posix(),
        "approved_segment_count": len(seed_records),
        "target_counts": target_counts,
        "validation": validation,
    }
    (output_root / "seed_package_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def audit_v7_segment_review_readiness(*, output_root: Path) -> dict[str, object]:
    """Audit manual review state without building seeds or changing approvals."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    pass_csv = output_root / "auto_quality_pass_segments.csv"
    packet_md = output_root / "segment_review_packet.md"
    contact_sheet = output_root / "review_contact_sheet.png"
    seed_npz = output_root / "v7_rps_seed_dataset.npz"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")
    records = {str(record["segment_id"]): record for record in _read_jsonl(proposed_path)}
    with review_path.open(encoding="utf-8") as handle:
        review_reader = csv.DictReader(handle)
        review_fieldnames = list(review_reader.fieldnames or [])
        review_rows = list(review_reader)
    approved_rows = [
        row
        for row in review_rows
        if _truthy(row.get("approved_for_training", "")) and str(row.get("review_status", "")).strip().lower() == "approved"
    ]
    failures: list[dict[str, object]] = []
    manifest_integrity = _audit_review_manifest_integrity(
        review_rows=review_rows,
        review_fieldnames=review_fieldnames,
        proposed_segment_ids=records.keys(),
    )
    failures.extend(cast(list[dict[str, object]], manifest_integrity.get("failures", [])))
    visual_audit = _audit_review_visual_artifacts(output_root=output_root, records=records.values())
    failures.extend(cast(list[dict[str, object]], visual_audit.get("failures", [])))
    table_audit = _audit_review_table_artifacts(
        output_root=output_root,
        proposed_segment_ids=records.keys(),
        auto_quality_pass_segment_ids=(
            str(record.get("segment_id", "")).strip()
            for record in records.values()
            if record.get("quality_status") == "auto_quality_pass"
        ),
        enforce_decision_template_hashes=not approved_rows,
    )
    failures.extend(cast(list[dict[str, object]], table_audit.get("failures", [])))
    approved_records: list[Mapping[str, object]] = []
    for row in approved_rows:
        segment_id = str(row.get("segment_id", ""))
        record = records.get(segment_id)
        if record is None:
            failures.append({"code": "approved_segment_missing_from_proposals", "segment_id": segment_id})
            continue
        if record.get("quality_status") != "auto_quality_pass":
            failures.append({"code": "approved_segment_failed_auto_quality", "segment_id": segment_id})
        source_path = str(record.get("source_path", ""))
        _reject_heldout_path(source_path, proposed_path)
        skeleton_npz = _resolve_artifact_path(record.get("skeleton_npz", ""), base=output_root)
        if not skeleton_npz.exists():
            failures.append({"code": "approved_segment_missing_skeleton_npz", "segment_id": segment_id, "skeleton_npz": skeleton_npz.as_posix()})
        approved_records.append(record)
    quality_counts = Counter(str(record.get("quality_status", "")) for record in records.values())
    eligible_records = [record for record in records.values() if record.get("quality_status") == "auto_quality_pass"]
    approved_quality_pass_count = sum(1 for record in approved_records if record.get("quality_status") == "auto_quality_pass")
    eligible_target_counts = Counter(str(record.get("target_name", "")) for record in eligible_records)
    target_counts = Counter(str(record.get("target_name", "")) for record in approved_records)
    missing_approved_targets = [target_name for target_name in TARGET_NAMES if int(target_counts.get(target_name, 0)) == 0]
    approved_segments_missing_review_notes = sorted(
        str(row.get("segment_id", "")).strip()
        for row in approved_rows
        if str(row.get("segment_id", "")).strip() and not str(row.get("review_notes", "")).strip()
    )
    warnings: list[dict[str, object]] = []
    if approved_rows and missing_approved_targets:
        warnings.append(
            {
                "code": "approved_seed_class_coverage_incomplete",
                "missing_targets": missing_approved_targets,
                "policy": "allowed before v7_rps_pose expansion; generator must add matched procedural controls to keep final shards balanced",
            }
        )
    if approved_segments_missing_review_notes:
        warnings.append(
            {
                "code": "approved_segment_missing_review_notes",
                "segment_ids": approved_segments_missing_review_notes,
                "policy": "non-blocking audit warning; add review_notes before seed packaging when possible",
            }
        )
    failure_codes = {str(failure.get("code", "")) for failure in failures}
    status = (
        "invalid_review_artifacts"
        if any(code.startswith("review_") for code in failure_codes)
        else
        "invalid_review_manifest"
        if failures
        else "ready_for_seed_package_build"
        if approved_rows
        else "awaiting_manual_segment_approval"
    )
    summary: dict[str, object] = {
        "status": status,
        "output_root": output_root.as_posix(),
        "proposed_segment_count": len(records),
        "eligible_quality_pass_count": int(quality_counts.get("auto_quality_pass", 0)),
        "auto_quality_failed_count": int(quality_counts.get("auto_quality_failed", 0)),
        "approved_segment_count": len(approved_rows),
        "approved_quality_pass_count": approved_quality_pass_count,
        "eligible_target_counts": dict(sorted(eligible_target_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "missing_approved_targets": missing_approved_targets if approved_rows else [],
        "approved_segments_missing_review_notes": approved_segments_missing_review_notes,
        "warnings": warnings,
        "seed_npz_exists": seed_npz.exists(),
        "review_manifest": review_path.as_posix(),
        "proposed_segments": proposed_path.as_posix(),
        "auto_quality_pass_segments": pass_csv.as_posix() if pass_csv.exists() else None,
        "segment_review_packet": packet_md.as_posix() if packet_md.exists() else None,
        "review_contact_sheet": contact_sheet.as_posix() if contact_sheet.exists() else None,
        "review_manifest_integrity": manifest_integrity,
        "review_visual_artifacts": visual_audit,
        "review_table_artifacts": table_audit,
        "failures": failures,
        "next_action": _review_next_action(status),
    }
    write_v7_review_readiness_report(output_root=output_root, summary=summary)
    return summary


def write_v7_segment_review_worklist(*, output_root: Path) -> dict[str, object]:
    """Write a reviewer worklist without changing the manual approval manifest."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")
    records = _read_jsonl(proposed_path)
    review_rows = {str(row.get("segment_id", "")): row for row in csv.DictReader(review_path.open(encoding="utf-8"))}
    worklist_rows: list[dict[str, object]] = []
    for record in records:
        segment_id = str(record.get("segment_id", ""))
        review_row = review_rows.get(segment_id, {})
        quality_status = str(record.get("quality_status", ""))
        eligible = quality_status == "auto_quality_pass"
        worklist_rows.append(
            {
                "review_priority": _review_worklist_priority(record, eligible=eligible),
                "segment_id": segment_id,
                "target_name": str(record.get("target_name", "")),
                "source_run_id": str(record.get("source_run_id", "")),
                "proposal_role": str(record.get("proposal_role", "")),
                "eligible_for_manual_approval": str(eligible).lower(),
                "quality_status": quality_status,
                "current_review_status": str(review_row.get("review_status", record.get("review_status", ""))),
                "currently_approved_for_training": str(
                    review_row.get("approved_for_training", record.get("approved_for_training", "false"))
                ).lower(),
                "frame_count": int(record.get("frame_count", 0)),
                "detection_coverage": f"{float(record.get('detection_coverage', 0.0)):.6f}",
                "severe_landmark_jump_count": int(record.get("severe_landmark_jump_count", 0)),
                "start_s": f"{float(record.get('start_s', 0.0)):.6f}",
                "end_s": f"{float(record.get('end_s', 0.0)):.6f}",
                "preview_image": _display_path(record.get("preview_image", ""), base=output_root),
                "skeleton_npz": _display_path(record.get("skeleton_npz", ""), base=output_root),
                "review_instruction": _review_worklist_instruction(record, eligible=eligible),
            }
        )
    worklist_rows.sort(
        key=lambda row: (
            0 if row["eligible_for_manual_approval"] == "true" else 1,
            int(row["review_priority"]),
            str(row["source_run_id"]),
            str(row["segment_id"]),
        )
    )
    csv_path = output_root / "segment_review_worklist.csv"
    md_path = output_root / "segment_review_worklist.md"
    html_path = output_root / "segment_review_gallery.html"
    _write_segment_review_worklist_csv(csv_path, worklist_rows)
    _write_segment_review_worklist_md(md_path, worklist_rows, output_root=output_root)
    _write_segment_review_gallery_html(html_path, worklist_rows, output_root=output_root)
    eligible_count = sum(1 for row in worklist_rows if row["eligible_for_manual_approval"] == "true")
    approved_count = sum(
        1
        for row in worklist_rows
        if _truthy(str(row["currently_approved_for_training"])) and str(row["current_review_status"]).lower() == "approved"
    )
    summary = {
        "status": "worklist_written",
        "output_root": output_root.as_posix(),
        "worklist_csv": csv_path.as_posix(),
        "worklist_md": md_path.as_posix(),
        "worklist_html": html_path.as_posix(),
        "proposed_segment_count": len(worklist_rows),
        "eligible_review_count": eligible_count,
        "ineligible_review_count": len(worklist_rows) - eligible_count,
        "approved_segment_count": approved_count,
        "approval_manifest_unchanged": review_path.as_posix(),
        "next_action": "inspect segment_review_worklist.md and edit segment_review_manifest.csv only for visually approved rows",
    }
    (output_root / "segment_review_worklist_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def write_v7_segment_review_coverage_report(*, output_root: Path) -> dict[str, object]:
    """Write target-coverage review aids without changing approvals."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")

    records = _read_jsonl(proposed_path)
    review_rows = {str(row.get("segment_id", "")): row for row in csv.DictReader(review_path.open(encoding="utf-8"))}
    coverage: dict[str, dict[str, object]] = {
        target_name: {
            "eligible_count": 0,
            "ineligible_count": 0,
            "approved_count": 0,
            "approved_quality_pass_count": 0,
            "eligible_segment_ids": [],
            "ineligible_segment_ids": [],
            "approved_segment_ids": [],
        }
        for target_name in TARGET_NAMES
    }

    for record in records:
        target_name = str(record.get("target_name", ""))
        if target_name not in coverage:
            continue
        segment_id = str(record.get("segment_id", ""))
        target_coverage = coverage[target_name]
        eligible = str(record.get("quality_status", "")) == "auto_quality_pass"
        review_row = review_rows.get(segment_id, {})
        approved = _truthy(review_row.get("approved_for_training", "")) and str(
            review_row.get("review_status", "")
        ).strip().lower() == "approved"
        if eligible:
            target_coverage["eligible_count"] = int(target_coverage["eligible_count"]) + 1
            cast(list[str], target_coverage["eligible_segment_ids"]).append(segment_id)
        else:
            target_coverage["ineligible_count"] = int(target_coverage["ineligible_count"]) + 1
            cast(list[str], target_coverage["ineligible_segment_ids"]).append(segment_id)
        if approved:
            target_coverage["approved_count"] = int(target_coverage["approved_count"]) + 1
            cast(list[str], target_coverage["approved_segment_ids"]).append(segment_id)
            if eligible:
                target_coverage["approved_quality_pass_count"] = int(target_coverage["approved_quality_pass_count"]) + 1

    missing_eligible_targets = [
        target_name for target_name in TARGET_NAMES if int(coverage[target_name]["eligible_count"]) == 0
    ]
    approved_segment_count = sum(int(item["approved_count"]) for item in coverage.values())
    missing_approved_targets = [
        target_name
        for target_name in TARGET_NAMES
        if approved_segment_count and int(coverage[target_name]["approved_count"]) == 0
    ]
    warnings: list[dict[str, object]] = [
        {
            "code": "no_eligible_real_review_rows_for_target",
            "target_name": target_name,
            "policy": (
                "Do not approve failed-quality rows; rely on approved seed packages or matched procedural controls "
                "unless new quality-passed real evidence is added."
            ),
        }
        for target_name in missing_eligible_targets
    ]
    if missing_approved_targets:
        warnings.append(
            {
                "code": "approved_seed_class_coverage_incomplete",
                "missing_targets": missing_approved_targets,
                "policy": (
                    "Allowed before v7_rps_pose expansion; generator must add matched procedural controls to keep "
                    "final shards balanced."
                ),
            }
        )

    summary: dict[str, object] = {
        "status": "coverage_report_written",
        "output_root": _display_path(output_root, base=output_root),
        "proposed_segment_count": len(records),
        "approved_segment_count": approved_segment_count,
        "target_coverage": coverage,
        "missing_eligible_targets": missing_eligible_targets,
        "missing_approved_targets": missing_approved_targets,
        "warnings": warnings,
        "review_manifest_unchanged": _display_path(review_path, base=output_root),
        "next_action": "review quality-passed candidates by target; do not approve failed-quality rows",
    }
    json_path = output_root / "segment_review_coverage_summary.json"
    md_path = output_root / "segment_review_coverage_summary.md"
    json_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_v7_segment_review_coverage_md(md_path, summary=summary)
    return summary


def _write_v7_segment_review_coverage_md(path: Path, *, summary: Mapping[str, object]) -> Path:
    coverage = _mapping(summary.get("target_coverage", {}))
    lines = [
        "# V7 Segment Review Target Coverage",
        "",
        "This report is a review aid. It does not approve segments, build the seed NPZ, generate the expanded dataset, train, validate, or promote v7.",
        "",
        "## Target Coverage",
        "",
        "| Target | Eligible | Ineligible | Approved | Approved Quality-Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for target_name in TARGET_NAMES:
        row = _mapping(coverage.get(target_name, {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    target_name,
                    str(row.get("eligible_count", 0)),
                    str(row.get("ineligible_count", 0)),
                    str(row.get("approved_count", 0)),
                    str(row.get("approved_quality_pass_count", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Review Rule",
            "",
            "Do not approve failed-quality rows. Approval remains limited to `approved_for_training = true` and `review_status = approved` in `segment_review_manifest.csv` after visual review.",
            "",
            f"- missing eligible targets: `{summary.get('missing_eligible_targets', [])}`",
            f"- missing approved targets: `{summary.get('missing_approved_targets', [])}`",
            f"- next action: `{summary.get('next_action')}`",
        ]
    )
    warnings = summary.get("warnings", [])
    if isinstance(warnings, Sequence) and warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_v7_segment_review_decision_template(*, output_root: Path) -> dict[str, object]:
    """Write a blank explicit-decision CSV for later manual segment approval."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")
    proposed_sha256 = _file_sha256(proposed_path)
    review_sha256 = _file_sha256(review_path)
    records = _sorted_review_records(_read_jsonl(proposed_path))
    review_rows = {str(row.get("segment_id", "")): row for row in csv.DictReader(review_path.open(encoding="utf-8"))}
    template_rows: list[dict[str, object]] = []
    for record in records:
        segment_id = str(record.get("segment_id", ""))
        review_row = review_rows.get(segment_id, {})
        eligible = str(record.get("quality_status", "")) == "auto_quality_pass"
        template_rows.append(
            {
                "segment_id": segment_id,
                "target_name": str(record.get("target_name", "")),
                "source_run_id": str(record.get("source_run_id", "")),
                "proposal_role": str(record.get("proposal_role", "")),
                "eligible_for_manual_approval": str(eligible).lower(),
                "quality_status": str(record.get("quality_status", "")),
                "current_review_status": str(review_row.get("review_status", record.get("review_status", ""))),
                "currently_approved_for_training": str(
                    review_row.get("approved_for_training", record.get("approved_for_training", "false"))
                ).lower(),
                "decision": "",
                "review_notes": "",
                "proposed_segments_sha256": proposed_sha256,
                "review_manifest_sha256": review_sha256,
                "preview_image": _display_path(record.get("preview_image", ""), base=output_root),
                "skeleton_npz": _display_path(record.get("skeleton_npz", ""), base=output_root),
                "instruction": "set decision to approve, reject, or needs_review after visual inspection",
            }
        )
    csv_path = output_root / "segment_review_decision_template.csv"
    md_path = output_root / "segment_review_decision_template.md"
    _write_segment_review_decision_template_csv(csv_path, template_rows)
    _write_segment_review_decision_template_md(md_path, template_rows)
    summary = {
        "status": "decision_template_written",
        "output_root": output_root.as_posix(),
        "decision_template_csv": csv_path.as_posix(),
        "decision_template_md": md_path.as_posix(),
        "proposed_segment_count": len(template_rows),
        "eligible_review_count": sum(1 for row in template_rows if row["eligible_for_manual_approval"] == "true"),
        "decision_rows_populated": 0,
        "proposed_segments_sha256": proposed_sha256,
        "review_manifest_sha256": review_sha256,
        "approval_manifest_unchanged": review_path.as_posix(),
        "next_action": "fill explicit decisions in segment_review_decision_template.csv, then dry-run apply_v7_segment_review_decisions",
    }
    (output_root / "segment_review_decision_template_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def apply_v7_segment_review_decisions(*, output_root: Path, decisions_csv: Path, apply: bool = False) -> dict[str, object]:
    """Validate or apply explicit segment review decisions to the review manifest."""

    proposed_path = output_root / "proposed_segments.jsonl"
    review_path = output_root / "segment_review_manifest.csv"
    if not proposed_path.exists():
        raise FileNotFoundError(f"Missing proposed segments: {proposed_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review manifest: {review_path}")
    if not decisions_csv.exists():
        raise FileNotFoundError(f"Missing decisions CSV: {decisions_csv}")
    proposed_sha256 = _file_sha256(proposed_path)
    review_sha256 = _file_sha256(review_path)

    records = {str(record.get("segment_id", "")): record for record in _read_jsonl(proposed_path)}
    with review_path.open(encoding="utf-8") as handle:
        review_reader = csv.DictReader(handle)
        review_fieldnames = list(review_reader.fieldnames or [])
        review_rows = list(review_reader)
    if not review_fieldnames:
        raise ValueError(f"{review_path} has no header")
    if "segment_id" not in review_fieldnames or "approved_for_training" not in review_fieldnames or "review_status" not in review_fieldnames:
        raise ValueError(f"{review_path} is missing required review columns")
    review_by_id = {str(row.get("segment_id", "")): row for row in review_rows}
    with decisions_csv.open(encoding="utf-8") as handle:
        decision_reader = csv.DictReader(handle)
        decision_fieldnames = set(decision_reader.fieldnames or [])
        decision_rows = list(decision_reader)
    decisions: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    required_decision_columns = {"segment_id", "decision", "proposed_segments_sha256", "review_manifest_sha256"}
    missing_decision_columns = sorted(required_decision_columns.difference(decision_fieldnames))
    if missing_decision_columns:
        failures.append({"code": "decision_csv_missing_required_columns", "columns": missing_decision_columns})
    populated_segment_ids: set[str] = set()
    for row_index, row in enumerate(decision_rows, start=2):
        segment_id = str(row.get("segment_id", "")).strip()
        raw_decision = str(row.get("decision", "")).strip()
        if not raw_decision:
            continue
        if segment_id in populated_segment_ids:
            failures.append({"code": "duplicate_review_decision_segment", "segment_id": segment_id, "row": row_index})
            continue
        if segment_id:
            populated_segment_ids.add(segment_id)
        if str(row.get("proposed_segments_sha256", "")).strip() != proposed_sha256:
            failures.append({"code": "stale_proposed_segments_hash", "segment_id": segment_id, "row": row_index})
            continue
        if str(row.get("review_manifest_sha256", "")).strip() != review_sha256:
            failures.append({"code": "stale_review_manifest_hash", "segment_id": segment_id, "row": row_index})
            continue
        decision = _normalize_review_decision(raw_decision)
        if not segment_id:
            failures.append({"code": "missing_segment_id", "row": row_index})
            continue
        record = records.get(segment_id)
        if record is None:
            failures.append({"code": "decision_segment_missing_from_proposals", "segment_id": segment_id, "row": row_index})
            continue
        if segment_id not in review_by_id:
            failures.append({"code": "decision_segment_missing_from_review_manifest", "segment_id": segment_id, "row": row_index})
            continue
        if decision is None:
            failures.append({"code": "unsupported_review_decision", "segment_id": segment_id, "decision": raw_decision, "row": row_index})
            continue
        if decision == "approve":
            if record.get("quality_status") != "auto_quality_pass":
                failures.append({"code": "cannot_approve_failed_auto_quality", "segment_id": segment_id, "row": row_index})
            if _is_heldout_path(str(record.get("source_path", ""))):
                failures.append({"code": "cannot_approve_heldout_source_path", "segment_id": segment_id, "row": row_index})
            skeleton_npz = _resolve_artifact_path(record.get("skeleton_npz", ""), base=output_root)
            if not skeleton_npz.exists():
                failures.append({"code": "cannot_approve_missing_skeleton_npz", "segment_id": segment_id, "row": row_index})
        decisions.append(
            {
                "segment_id": segment_id,
                "decision": decision,
                "target_name": str(record.get("target_name", "")),
                "review_notes": str(row.get("review_notes", "")).strip(),
                "row": row_index,
            }
        )

    applied_count = 0
    if not failures and apply and decisions:
        for decision in decisions:
            review_row = review_by_id[str(decision["segment_id"])]
            normalized = str(decision["decision"])
            if normalized == "approve":
                review_row["approved_for_training"] = "true"
                review_row["review_status"] = "approved"
            elif normalized == "reject":
                review_row["approved_for_training"] = "false"
                review_row["review_status"] = "rejected"
            else:
                review_row["approved_for_training"] = "false"
                review_row["review_status"] = "pending_manual_review"
            if "review_notes" in review_fieldnames and str(decision.get("review_notes", "")):
                review_row["review_notes"] = str(decision["review_notes"])
            applied_count += 1
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=review_fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)
        write_v7_segment_review_coverage_report(output_root=output_root)

    approve_count = sum(1 for decision in decisions if decision.get("decision") == "approve")
    reject_count = sum(1 for decision in decisions if decision.get("decision") == "reject")
    needs_review_count = sum(1 for decision in decisions if decision.get("decision") == "needs_review")
    decision_target_counts = _decision_target_counts(decisions)
    approve_target_counts = _decision_target_counts(decisions, decision_name="approve")
    reject_target_counts = _decision_target_counts(decisions, decision_name="reject")
    needs_review_target_counts = _decision_target_counts(decisions, decision_name="needs_review")
    approve_segment_ids_by_target = _decision_segment_ids_by_target(decisions, decision_name="approve")
    missing_approve_targets = _missing_approve_targets(approve_target_counts)
    approval_decisions_missing_review_notes = _approval_decisions_missing_review_notes(decisions)
    warnings = []
    if missing_approve_targets:
        warnings.append("approved_decision_class_coverage_incomplete")
    if approval_decisions_missing_review_notes:
        warnings.append("approval_decision_missing_review_notes")
    status = (
        "invalid_decisions"
        if failures
        else "applied"
        if apply and decisions
        else "dry_run_ready"
        if decisions
        else "no_review_decisions"
    )
    summary = {
        "status": status,
        "apply": bool(apply),
        "output_root": output_root.as_posix(),
        "decisions_csv": decisions_csv.as_posix(),
        "review_manifest": review_path.as_posix(),
        "current_proposed_segments_sha256": proposed_sha256,
        "current_review_manifest_sha256": review_sha256,
        "decision_count": len(decisions),
        "approve_count": approve_count,
        "reject_count": reject_count,
        "needs_review_count": needs_review_count,
        "decision_target_counts": decision_target_counts,
        "approve_target_counts": approve_target_counts,
        "reject_target_counts": reject_target_counts,
        "needs_review_target_counts": needs_review_target_counts,
        "approve_segment_ids_by_target": approve_segment_ids_by_target,
        "missing_approve_targets": missing_approve_targets,
        "approval_decisions_missing_review_notes": approval_decisions_missing_review_notes,
        "warnings": warnings,
        "applied_count": applied_count,
        "failures": failures,
        "next_action": _review_decision_next_action(status),
    }
    (output_root / "segment_review_decision_apply_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_segment_review_decision_apply_summary_md(
        output_root / "segment_review_decision_apply_summary.md",
        summary,
    )
    return summary


def _write_segment_review_decision_apply_summary_md(path: Path, summary: Mapping[str, object]) -> Path:
    lines = [
        "# V7 Segment Review Decision Apply Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- apply: `{summary.get('apply')}`",
        f"- decision count: `{summary.get('decision_count')}`",
        f"- approve count: `{summary.get('approve_count')}`",
        f"- reject count: `{summary.get('reject_count')}`",
        f"- needs-review count: `{summary.get('needs_review_count')}`",
        f"- applied count: `{summary.get('applied_count')}`",
        f"- missing approve targets: `{summary.get('missing_approve_targets', [])}`",
        f"- approval decisions missing review notes: `{summary.get('approval_decisions_missing_review_notes', [])}`",
        f"- next action: `{summary.get('next_action')}`",
        "",
        "## Target Counts",
        "",
        f"- decisions: `{summary.get('decision_target_counts', {})}`",
        f"- approvals: `{summary.get('approve_target_counts', {})}`",
        f"- rejections: `{summary.get('reject_target_counts', {})}`",
        f"- needs review: `{summary.get('needs_review_target_counts', {})}`",
        "",
        "## Approved Segment IDs By Target",
        "",
    ]
    segment_ids_by_target = summary.get("approve_segment_ids_by_target", {})
    if isinstance(segment_ids_by_target, Mapping) and segment_ids_by_target:
        for target_name, segment_ids in sorted(segment_ids_by_target.items()):
            lines.append(f"- `{target_name}`: `{segment_ids}`")
    else:
        lines.append("- none")
    warnings = summary.get("warnings", [])
    if isinstance(warnings, Sequence) and warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning}`")
    failures = summary.get("failures", [])
    if isinstance(failures, Sequence) and failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _missing_approve_targets(approve_target_counts: Mapping[str, int]) -> list[str]:
    if not approve_target_counts:
        return []
    return sorted(target_name for target_name in TARGET_NAMES if int(approve_target_counts.get(target_name, 0)) <= 0)


def _approval_decisions_missing_review_notes(decisions: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        str(decision.get("segment_id", "")).strip()
        for decision in decisions
        if str(decision.get("decision", "")) == "approve"
        and not str(decision.get("review_notes", "")).strip()
        and str(decision.get("segment_id", "")).strip()
    )


def _decision_target_counts(
    decisions: Sequence[Mapping[str, object]],
    *,
    decision_name: str | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        if decision_name is not None and str(decision.get("decision", "")) != decision_name:
            continue
        target_name = str(decision.get("target_name", "")).strip()
        if target_name:
            counts[target_name] += 1
    return dict(sorted(counts.items()))


def _decision_segment_ids_by_target(
    decisions: Sequence[Mapping[str, object]],
    *,
    decision_name: str,
) -> dict[str, list[str]]:
    segment_ids_by_target: dict[str, list[str]] = {}
    for decision in decisions:
        if str(decision.get("decision", "")) != decision_name:
            continue
        target_name = str(decision.get("target_name", "")).strip()
        segment_id = str(decision.get("segment_id", "")).strip()
        if not target_name or not segment_id:
            continue
        segment_ids_by_target.setdefault(target_name, []).append(segment_id)
    return {target_name: sorted(segment_ids) for target_name, segment_ids in sorted(segment_ids_by_target.items())}


def _sorted_review_records(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        records,
        key=lambda record: (
            0 if record.get("quality_status") == "auto_quality_pass" else 1,
            _review_worklist_priority(record, eligible=record.get("quality_status") == "auto_quality_pass"),
            str(record.get("source_run_id", "")),
            str(record.get("segment_id", "")),
        ),
    )


def _write_segment_review_decision_template_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    fieldnames = [
        "segment_id",
        "target_name",
        "source_run_id",
        "proposal_role",
        "eligible_for_manual_approval",
        "quality_status",
        "current_review_status",
        "currently_approved_for_training",
        "decision",
        "review_notes",
        "preview_image",
        "skeleton_npz",
        "proposed_segments_sha256",
        "review_manifest_sha256",
        "instruction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_segment_review_decision_template_md(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    eligible_count = sum(1 for row in rows if row.get("eligible_for_manual_approval") == "true")
    lines = [
        "# V7 Segment Review Decision Template",
        "",
        "Fill the `decision` column in `segment_review_decision_template.csv` only after visual review.",
        "",
        "Allowed decisions:",
        "",
        "- `approve`: mark the segment as approved for training after validation and explicit apply.",
        "- `reject`: keep the segment out of training.",
        "- `needs_review`: keep the segment pending.",
        "- blank: leave the current manifest row unchanged.",
        "",
        "Apply is never implicit. First run the apply command without `--apply`; rerun with `--apply` only after the dry run reports valid decisions.",
        "",
        "The template includes SHA-256 provenance columns for `proposed_segments.jsonl` and `segment_review_manifest.csv`. If either source changes after the template is generated, the apply command rejects populated decisions until a fresh template is written.",
        "",
        "## Summary",
        "",
        f"- proposed rows: `{len(rows)}`",
        f"- eligible approval rows: `{eligible_count}`",
        f"- ineligible rows: `{len(rows) - eligible_count}`",
        "",
        "## Commands",
        "",
        "```text",
        "python -m embodied_rps.tools.apply_v7_segment_review_decisions",
        "python -m embodied_rps.tools.apply_v7_segment_review_decisions --apply",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _normalize_review_decision(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"approve", "approved", "accept", "accepted"}:
        return "approve"
    if normalized in {"reject", "rejected", "deny", "denied"}:
        return "reject"
    if normalized in {"needs_review", "pending", "pending_manual_review"}:
        return "needs_review"
    return None


def _review_decision_next_action(status: str) -> str:
    if status == "invalid_decisions":
        return "fix decision CSV failures before applying review decisions"
    if status == "applied":
        return "run python -m embodied_rps.tools.audit_v7_segment_review_readiness"
    if status == "dry_run_ready":
        return "rerun with --apply only if the reviewed decisions are intentional"
    return "fill explicit decisions in the decision CSV after visual review"


def _review_worklist_priority(record: Mapping[str, object], *, eligible: bool) -> int:
    if not eligible:
        return 90
    role = str(record.get("proposal_role", ""))
    if role == "scissors_transition":
        return 10
    if role == "scissors_static":
        return 20
    return 30


def _review_worklist_instruction(record: Mapping[str, object], *, eligible: bool) -> str:
    if eligible:
        return "inspect_preview_and_segment_npz_then_edit_segment_review_manifest_only_if_visually_correct"
    reasons: list[str] = ["do_not_approve_failed_auto_quality"]
    if float(record.get("detection_coverage", 0.0)) < 0.95:
        reasons.append("detection_coverage_below_0.95")
    if int(record.get("severe_landmark_jump_count", 0)) > 0:
        reasons.append("severe_landmark_jumps_present")
    if not bool(record.get("finite", True)):
        reasons.append("non_finite_landmarks")
    return ";".join(reasons)


def _audit_review_visual_artifacts(*, output_root: Path, records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    preview_paths: list[Path] = []
    missing_previews: list[str] = []
    for record in records:
        raw_preview = str(record.get("preview_image", "")).strip()
        if not raw_preview:
            continue
        preview_path = _resolve_artifact_path(raw_preview, base=output_root)
        if preview_path.exists():
            preview_paths.append(preview_path)
        else:
            missing_previews.append(raw_preview)

    contact_sheet = output_root / "review_contact_sheet.png"
    contact_sheet_info = _inspect_png(contact_sheet)
    failures: list[dict[str, object]] = []
    if missing_previews:
        failures.append(
            {
                "code": "review_preview_missing",
                "missing_count": len(missing_previews),
                "examples": missing_previews[:5],
            }
        )
    if preview_paths:
        if not contact_sheet.exists():
            failures.append({"code": "review_contact_sheet_missing", "path": contact_sheet.as_posix()})
        elif not bool(contact_sheet_info.get("valid_png", False)):
            failures.append({"code": "review_contact_sheet_invalid_png", "path": contact_sheet.as_posix()})
        elif int(contact_sheet_info.get("width", 0)) <= 1 or int(contact_sheet_info.get("height", 0)) <= 1:
            failures.append(
                {
                    "code": "review_contact_sheet_placeholder",
                    "path": contact_sheet.as_posix(),
                    "width": contact_sheet_info.get("width"),
                    "height": contact_sheet_info.get("height"),
                    "byte_size": contact_sheet_info.get("byte_size"),
                }
            )

    status = (
        "failed"
        if failures
        else "passed"
        if preview_paths
        else "not_applicable_no_preview_images"
    )
    return {
        "status": status,
        "preview_image_count": len(preview_paths) + len(missing_previews),
        "existing_preview_image_count": len(preview_paths),
        "missing_preview_image_count": len(missing_previews),
        "contact_sheet": _display_path(contact_sheet, base=output_root) if contact_sheet.exists() else None,
        "contact_sheet_info": contact_sheet_info,
        "failures": failures,
    }


def _audit_review_manifest_integrity(
    *,
    review_rows: Sequence[Mapping[str, object]],
    review_fieldnames: Sequence[str],
    proposed_segment_ids: Iterable[str],
) -> dict[str, object]:
    required_columns = ("segment_id", "approved_for_training", "review_status", "review_notes")
    fieldname_set = {str(field) for field in review_fieldnames}
    missing_required_columns = sorted(set(required_columns).difference(fieldname_set))
    expected_ids = {str(segment_id).strip() for segment_id in proposed_segment_ids if str(segment_id).strip()}
    seen_counts = Counter(str(row.get("segment_id", "")).strip() for row in review_rows if str(row.get("segment_id", "")).strip())
    manifest_ids = set(seen_counts)
    missing_ids = sorted(expected_ids - manifest_ids)
    extra_ids = sorted(manifest_ids - expected_ids)
    duplicate_ids = sorted(segment_id for segment_id, count in seen_counts.items() if count > 1)
    valid_approval_values = {"false", "true"}
    valid_review_statuses = {"approved", "pending_manual_review", "rejected"}
    invalid_approval_value_rows = [
        {
            "segment_id": str(row.get("segment_id", "")).strip(),
            "approved_for_training": str(row.get("approved_for_training", "")).strip(),
        }
        for row in review_rows
        if "approved_for_training" in fieldname_set
        and str(row.get("segment_id", "")).strip()
        and str(row.get("approved_for_training", "")).strip().lower() not in valid_approval_values
    ]
    invalid_review_status_rows = [
        {
            "segment_id": str(row.get("segment_id", "")).strip(),
            "review_status": str(row.get("review_status", "")).strip(),
        }
        for row in review_rows
        if "review_status" in fieldname_set
        and str(row.get("segment_id", "")).strip()
        and str(row.get("review_status", "")).strip().lower() not in valid_review_statuses
    ]
    invalid_approval_value_segment_ids = sorted(str(row["segment_id"]) for row in invalid_approval_value_rows)
    invalid_review_status_segment_ids = sorted(str(row["segment_id"]) for row in invalid_review_status_rows)
    inconsistent_approval_rows = [
        {
            "segment_id": str(row.get("segment_id", "")).strip(),
            "approved_for_training": str(row.get("approved_for_training", "")).strip(),
            "review_status": str(row.get("review_status", "")).strip(),
        }
        for row in review_rows
        if str(row.get("segment_id", "")).strip()
        and (
            _truthy(row.get("approved_for_training", ""))
            != (str(row.get("review_status", "")).strip().lower() == "approved")
        )
    ]
    inconsistent_approval_segment_ids = sorted(str(row["segment_id"]) for row in inconsistent_approval_rows)
    failures: list[dict[str, object]] = []
    if missing_required_columns:
        failures.append(
            {
                "code": "segment_review_manifest_missing_required_columns",
                "missing_required_columns": missing_required_columns,
                "required_columns": list(required_columns),
            }
        )
    if missing_ids or extra_ids:
        failures.append(
            {
                "code": "segment_review_manifest_segment_ids_mismatch",
                "expected_segment_count": len(expected_ids),
                "row_count": len(review_rows),
                "missing_segment_ids": missing_ids,
                "extra_segment_ids": extra_ids,
            }
        )
    if duplicate_ids:
        failures.append(
            {
                "code": "segment_review_manifest_duplicate_segment_ids",
                "duplicate_segment_ids": duplicate_ids,
            }
        )
    if invalid_approval_value_rows:
        failures.append(
            {
                "code": "segment_review_manifest_invalid_approved_for_training",
                "rows": invalid_approval_value_rows,
                "valid_values": sorted(valid_approval_values),
            }
        )
    if invalid_review_status_rows:
        failures.append(
            {
                "code": "segment_review_manifest_invalid_review_status",
                "rows": invalid_review_status_rows,
                "valid_values": sorted(valid_review_statuses),
            }
        )
    if inconsistent_approval_rows:
        failures.append(
            {
                "code": "segment_review_manifest_inconsistent_approval_state",
                "rows": inconsistent_approval_rows,
                "policy": "approved_for_training=true is valid only when review_status=approved; review_status=approved is valid only when approved_for_training=true",
            }
        )
    status = (
        "missing_required_columns"
        if missing_required_columns
        else "duplicate_segment_ids"
        if duplicate_ids
        else "segment_id_mismatch"
        if missing_ids or extra_ids
        else "invalid_field_values"
        if invalid_approval_value_rows or invalid_review_status_rows
        else "inconsistent_approval_state"
        if inconsistent_approval_rows
        else "passed"
    )
    return {
        "status": status,
        "row_count": len(review_rows),
        "expected_segment_count": len(expected_ids),
        "required_columns": list(required_columns),
        "missing_required_columns": missing_required_columns,
        "missing_segment_ids": missing_ids,
        "extra_segment_ids": extra_ids,
        "duplicate_segment_ids": duplicate_ids,
        "valid_approval_values": sorted(valid_approval_values),
        "valid_review_statuses": sorted(valid_review_statuses),
        "invalid_approval_value_segment_ids": invalid_approval_value_segment_ids,
        "invalid_approval_value_rows": invalid_approval_value_rows,
        "invalid_review_status_segment_ids": invalid_review_status_segment_ids,
        "invalid_review_status_rows": invalid_review_status_rows,
        "inconsistent_approval_segment_ids": inconsistent_approval_segment_ids,
        "inconsistent_approval_rows": inconsistent_approval_rows,
        "failures": failures,
    }


def _audit_review_table_artifacts(
    *,
    output_root: Path,
    proposed_segment_ids: Iterable[str],
    auto_quality_pass_segment_ids: Iterable[str],
    enforce_decision_template_hashes: bool,
) -> dict[str, object]:
    proposed_ids = {str(segment_id).strip() for segment_id in proposed_segment_ids if str(segment_id).strip()}
    pass_ids = {str(segment_id).strip() for segment_id in auto_quality_pass_segment_ids if str(segment_id).strip()}
    failures: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    expected_hashes = {
        "proposed_segments_sha256": _file_sha256(output_root / "proposed_segments.jsonl"),
        "review_manifest_sha256": _file_sha256(output_root / "segment_review_manifest.csv"),
    }
    for artifact_name, filename in (
        ("auto_quality_pass_segments", "auto_quality_pass_segments.csv"),
        ("worklist", "segment_review_worklist.csv"),
        ("decision_template", "segment_review_decision_template.csv"),
    ):
        path = output_root / filename
        artifact = _audit_review_segment_id_csv(
            path=path,
            artifact_name=artifact_name,
            expected_ids=pass_ids if artifact_name == "auto_quality_pass_segments" else proposed_ids,
            output_root=output_root,
        )
        if artifact_name == "decision_template" and bool(artifact.get("exists")):
            _audit_decision_template_hashes(
                artifact=artifact,
                path=path,
                rows=cast(list[dict[str, object]], artifact.get("rows", [])),
                expected_hashes=expected_hashes,
                enforce_hashes=enforce_decision_template_hashes,
                output_root=output_root,
            )
        artifact.pop("rows", None)
        artifacts[artifact_name] = artifact
        failures.extend(cast(list[dict[str, object]], artifact.get("failures", [])))

    gallery = _audit_review_gallery_html(
        path=output_root / "segment_review_gallery.html",
        expected_ids=pass_ids,
        output_root=output_root,
        required=bool(artifacts["worklist"].get("exists")),
    )
    artifacts["gallery"] = gallery
    failures.extend(cast(list[dict[str, object]], gallery.get("failures", [])))

    return {
        "status": "failed" if failures else "passed" if any(artifact.get("exists") for artifact in artifacts.values()) else "not_applicable_no_review_tables",
        "auto_quality_pass_segments": artifacts["auto_quality_pass_segments"],
        "worklist": artifacts["worklist"],
        "gallery": artifacts["gallery"],
        "decision_template": artifacts["decision_template"],
        "failures": failures,
    }


def _audit_review_segment_id_csv(
    *,
    path: Path,
    artifact_name: str,
    expected_ids: set[str],
    output_root: Path,
) -> dict[str, object]:
    if not path.exists():
        return {
            "status": "missing",
            "exists": False,
            "path": _display_path(path, base=output_root),
            "row_count": 0,
            "missing_segment_ids": [],
            "extra_segment_ids": [],
            "rows": [],
            "failures": [],
        }
    rows = cast(list[dict[str, object]], list(csv.DictReader(path.open(encoding="utf-8"))))
    artifact_ids = {str(row.get("segment_id", "")).strip() for row in rows if str(row.get("segment_id", "")).strip()}
    missing_ids = sorted(expected_ids - artifact_ids)
    extra_ids = sorted(artifact_ids - expected_ids)
    failures: list[dict[str, object]] = []
    if missing_ids or extra_ids:
        failures.append(
            {
                "code": f"review_{artifact_name}_segment_ids_mismatch",
                "path": _display_path(path, base=output_root),
                "expected_segment_count": len(expected_ids),
                "row_count": len(rows),
                "missing_segment_ids": missing_ids,
                "extra_segment_ids": extra_ids,
            }
        )
    return {
        "status": "segment_id_mismatch" if failures else "passed",
        "exists": True,
        "path": _display_path(path, base=output_root),
        "row_count": len(rows),
        "missing_segment_ids": missing_ids,
        "extra_segment_ids": extra_ids,
        "rows": rows,
        "failures": failures,
    }


def _audit_decision_template_hashes(
    *,
    artifact: dict[str, object],
    path: Path,
    rows: Sequence[Mapping[str, object]],
    expected_hashes: Mapping[str, str],
    enforce_hashes: bool,
    output_root: Path,
) -> None:
    hash_columns = ("proposed_segments_sha256", "review_manifest_sha256")
    missing_columns = [column for column in hash_columns if rows and column not in rows[0]]
    failures = cast(list[dict[str, object]], artifact.get("failures", []))
    if missing_columns:
        failures.append(
            {
                "code": "review_decision_template_missing_hash_columns",
                "path": _display_path(path, base=output_root),
                "columns": missing_columns,
            }
        )
    else:
        mismatches: dict[str, dict[str, object]] = {}
        for column in hash_columns:
            observed = sorted({str(row.get(column, "")).strip() for row in rows})
            expected = str(expected_hashes[column])
            if observed != [expected]:
                mismatches[column] = {
                    "expected": expected,
                    "observed": observed,
                }
        if mismatches:
            mismatch = {
                "code": "review_decision_template_hash_mismatch",
                "path": _display_path(path, base=output_root),
                "mismatches": mismatches,
            }
            artifact["hash_mismatch"] = mismatch
            if enforce_hashes:
                failures.append(mismatch)
    artifact["failures"] = failures
    if failures:
        artifact["status"] = "hash_mismatch" if any(failure.get("code") == "review_decision_template_hash_mismatch" for failure in failures) else "missing_hash_columns"


def _audit_review_gallery_html(
    *,
    path: Path,
    expected_ids: set[str],
    output_root: Path,
    required: bool,
) -> dict[str, object]:
    if not path.exists():
        failures = (
            [
                {
                    "code": "review_gallery_missing",
                    "path": _display_path(path, base=output_root),
                    "policy": "refresh the v7 worklist so the browser gallery matches the current review packet",
                }
            ]
            if required
            else []
        )
        return {
            "status": "missing_required" if failures else "missing",
            "exists": False,
            "path": _display_path(path, base=output_root),
            "row_count": 0,
            "missing_segment_ids": [],
            "extra_segment_ids": [],
            "empty_src_count": 0,
            "failures": failures,
        }

    content = path.read_text(encoding="utf-8")
    gallery_ids = {
        html.unescape(match.group(1)).strip()
        for match in re.finditer(r'data-segment-id="([^"]+)"', content)
        if html.unescape(match.group(1)).strip()
    }
    missing_ids = sorted(expected_ids - gallery_ids)
    extra_ids = sorted(gallery_ids - expected_ids)
    empty_src_count = len(re.findall(r'<img\b[^>]*\bsrc=""', content))
    failures: list[dict[str, object]] = []
    if missing_ids or extra_ids:
        failures.append(
            {
                "code": "review_gallery_segment_ids_mismatch",
                "path": _display_path(path, base=output_root),
                "expected_segment_count": len(expected_ids),
                "row_count": len(gallery_ids),
                "missing_segment_ids": missing_ids,
                "extra_segment_ids": extra_ids,
            }
        )
    if empty_src_count:
        failures.append(
            {
                "code": "review_gallery_empty_image_src",
                "path": _display_path(path, base=output_root),
                "empty_src_count": empty_src_count,
            }
        )
    return {
        "status": "failed" if failures else "passed",
        "exists": True,
        "path": _display_path(path, base=output_root),
        "row_count": len(gallery_ids),
        "missing_segment_ids": missing_ids,
        "extra_segment_ids": extra_ids,
        "empty_src_count": empty_src_count,
        "failures": failures,
    }


def _inspect_png(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "valid_png": False, "width": 0, "height": 0, "byte_size": 0}
    data = path.read_bytes()
    info: dict[str, object] = {
        "exists": True,
        "valid_png": False,
        "width": 0,
        "height": 0,
        "byte_size": len(data),
    }
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
        info["valid_png"] = True
        info["width"] = int.from_bytes(data[16:20], "big")
        info["height"] = int.from_bytes(data[20:24], "big")
    return info


def _write_segment_review_worklist_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    fieldnames = [
        "review_priority",
        "segment_id",
        "target_name",
        "source_run_id",
        "proposal_role",
        "eligible_for_manual_approval",
        "quality_status",
        "current_review_status",
        "currently_approved_for_training",
        "frame_count",
        "detection_coverage",
        "severe_landmark_jump_count",
        "start_s",
        "end_s",
        "preview_image",
        "skeleton_npz",
        "review_instruction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_segment_review_worklist_md(path: Path, rows: Sequence[Mapping[str, object]], *, output_root: Path) -> Path:
    eligible_rows = [row for row in rows if row.get("eligible_for_manual_approval") == "true"]
    ineligible_rows = [row for row in rows if row.get("eligible_for_manual_approval") != "true"]
    lines = [
        "# V7 Segment Review Worklist",
        "",
        "This file is a review aid only. It does not approve segments and does not modify `segment_review_manifest.csv`.",
        "",
        "## Approval Contract",
        "",
        "Approve a segment only by editing `segment_review_manifest.csv` after visual review:",
        "",
        "```text",
        "approved_for_training = true",
        "review_status = approved",
        "```",
        "",
        "Do not approve rows marked `eligible_for_manual_approval = false`.",
        "",
        "## Summary",
        "",
        f"- proposed segments: `{len(rows)}`",
        f"- eligible for manual approval: `{len(eligible_rows)}`",
        f"- ineligible after automated quality checks: `{len(ineligible_rows)}`",
        "",
        "## Eligible Candidates",
        "",
        "| Priority | Segment | Run | Role | Detection | Preview |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in eligible_rows:
        preview_path = _display_path(row.get("preview_image", ""), base=output_root)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("review_priority", "")),
                    str(row.get("segment_id", "")),
                    str(row.get("source_run_id", "")),
                    str(row.get("proposal_role", "")),
                    str(row.get("detection_coverage", "")),
                    preview_path if preview_path else "not recorded",
                ]
            )
            + " |"
        )
    if eligible_rows:
        lines.extend(["", "## Eligible Preview Gallery", ""])
        for row in eligible_rows:
            segment_id = str(row.get("segment_id", ""))
            preview_path = _display_path(row.get("preview_image", ""), base=output_root)
            preview_block = f"![{segment_id}]({preview_path})" if preview_path else "`no preview image recorded`"
            lines.extend(
                [
                    f"### {segment_id}",
                    "",
                    preview_block,
                    "",
                    f"- target: `{row.get('target_name', '')}`",
                    f"- run: `{row.get('source_run_id', '')}`",
                    f"- role: `{row.get('proposal_role', '')}`",
                    f"- detection coverage: `{row.get('detection_coverage', '')}`",
                    f"- skeleton NPZ: `{_display_path(row.get('skeleton_npz', ''), base=output_root)}`",
                    "",
                ]
            )
    if ineligible_rows:
        lines.extend(["", "## Ineligible Candidates", "", "| Segment | Run | Reason |", "|---|---|---|"])
        for row in ineligible_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("segment_id", "")),
                        str(row.get("source_run_id", "")),
                        str(row.get("review_instruction", "")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_segment_review_gallery_html(path: Path, rows: Sequence[Mapping[str, object]], *, output_root: Path) -> Path:
    eligible_rows = [row for row in rows if row.get("eligible_for_manual_approval") == "true"]
    ineligible_rows = [row for row in rows if row.get("eligible_for_manual_approval") != "true"]
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>V7 Segment Review Gallery</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2328; }",
        "    .contract { border: 1px solid #d0d7de; padding: 12px; margin-bottom: 18px; background: #f6f8fa; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }",
        "    .card { border: 1px solid #d0d7de; padding: 12px; }",
        "    img { width: 100%; height: auto; display: block; border: 1px solid #d0d7de; }",
        "    .missing { padding: 32px 12px; border: 1px dashed #d0d7de; background: #f6f8fa; }",
        "    code { background: #f6f8fa; padding: 1px 4px; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #d0d7de; padding: 6px; text-align: left; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>V7 Segment Review Gallery</h1>",
        '  <div class="contract">',
        "    <p>This file is a review aid only. It does not approve segments and does not modify <code>segment_review_manifest.csv</code>.</p>",
        "    <p>Approve a segment only after visual review by setting <code>approved_for_training = true</code> and <code>review_status = approved</code> in <code>segment_review_manifest.csv</code>.</p>",
        "  </div>",
        f"  <p>Eligible candidates: <code>{len(eligible_rows)}</code>. Ineligible candidates: <code>{len(ineligible_rows)}</code>.</p>",
        "  <h2>Eligible Candidates</h2>",
        '  <div class="grid">',
    ]
    for row in eligible_rows:
        segment_id = _html_text(row.get("segment_id", ""))
        preview_path = _display_path(row.get("preview_image", ""), base=output_root)
        skeleton_npz = _display_path(row.get("skeleton_npz", ""), base=output_root)
        lines.extend(
            [
                f'    <section class="card" data-segment-id="{segment_id}">',
                f"      <h3>{segment_id}</h3>",
                _review_gallery_image_html(segment_id=segment_id, preview_path=preview_path),
                "      <ul>",
                f"        <li>target: <code>{_html_text(row.get('target_name', ''))}</code></li>",
                f"        <li>run: <code>{_html_text(row.get('source_run_id', ''))}</code></li>",
                f"        <li>role: <code>{_html_text(row.get('proposal_role', ''))}</code></li>",
                f"        <li>detection coverage: <code>{_html_text(row.get('detection_coverage', ''))}</code></li>",
                f"        <li>skeleton NPZ: <code>{_html_text(skeleton_npz)}</code></li>",
                "      </ul>",
                "    </section>",
            ]
        )
    lines.extend(["  </div>"])
    if ineligible_rows:
        lines.extend(
            [
                "  <h2>Ineligible Candidates</h2>",
                "  <table>",
                "    <thead><tr><th>Segment</th><th>Run</th><th>Reason</th></tr></thead>",
                "    <tbody>",
            ]
        )
        for row in ineligible_rows:
            lines.append(
                "      <tr>"
                f"<td>{_html_text(row.get('segment_id', ''))}</td>"
                f"<td>{_html_text(row.get('source_run_id', ''))}</td>"
                f"<td>{_html_text(row.get('review_instruction', ''))}</td>"
                "</tr>"
            )
        lines.extend(["    </tbody>", "  </table>"])
    lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _review_gallery_image_html(*, segment_id: str, preview_path: str) -> str:
    if not preview_path:
        return '      <div class="missing">no preview image recorded</div>'
    escaped_path = html.escape(preview_path, quote=True)
    return f'      <img src="{escaped_path}" alt="{segment_id}">'


def _html_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_v7_review_readiness_report(*, output_root: Path, summary: Mapping[str, object]) -> tuple[Path, Path]:
    json_path = output_root / "review_readiness_summary.json"
    md_path = output_root / "review_readiness_summary.md"
    json_path.write_text(json.dumps(_json_ready(dict(summary)), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V7 Review Readiness Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- proposed segments: `{summary.get('proposed_segment_count')}`",
        f"- auto-quality pass: `{summary.get('eligible_quality_pass_count')}`",
        f"- auto-quality failed: `{summary.get('auto_quality_failed_count')}`",
        f"- approved segments: `{summary.get('approved_segment_count')}`",
        f"- seed NPZ exists: `{summary.get('seed_npz_exists')}`",
        f"- next action: `{summary.get('next_action')}`",
        "",
        "## Review Files",
        "",
        f"- review manifest: `{_display_path(summary.get('review_manifest', ''), base=output_root)}`",
        f"- pass-only CSV: `{_display_path(summary.get('auto_quality_pass_segments', ''), base=output_root)}`",
        f"- review packet: `{_display_path(summary.get('segment_review_packet', ''), base=output_root)}`",
        f"- contact sheet: `{_display_path(summary.get('review_contact_sheet', ''), base=output_root)}`",
    ]
    lines.extend(
        [
            "",
            "## Target Coverage",
            "",
            f"- eligible target counts: `{summary.get('eligible_target_counts', {})}`",
            f"- approved target counts: `{summary.get('target_counts', {})}`",
            f"- missing approved targets: `{summary.get('missing_approved_targets', [])}`",
            f"- approved segments missing review notes: `{summary.get('approved_segments_missing_review_notes', [])}`",
        ]
    )
    manifest_integrity = summary.get("review_manifest_integrity", {})
    if isinstance(manifest_integrity, Mapping):
        lines.extend(
            [
                "",
                "## Review Manifest Integrity",
                "",
                f"- status: `{manifest_integrity.get('status')}`",
                f"- row count: `{manifest_integrity.get('row_count')}`",
                f"- missing segment IDs: `{manifest_integrity.get('missing_segment_ids', [])}`",
                f"- extra segment IDs: `{manifest_integrity.get('extra_segment_ids', [])}`",
                f"- duplicate segment IDs: `{manifest_integrity.get('duplicate_segment_ids', [])}`",
                f"- missing required columns: `{manifest_integrity.get('missing_required_columns', [])}`",
                f"- invalid approval-value segment IDs: `{manifest_integrity.get('invalid_approval_value_segment_ids', [])}`",
                f"- invalid review-status segment IDs: `{manifest_integrity.get('invalid_review_status_segment_ids', [])}`",
                f"- inconsistent approval segment IDs: `{manifest_integrity.get('inconsistent_approval_segment_ids', [])}`",
            ]
        )
    visual = summary.get("review_visual_artifacts", {})
    if isinstance(visual, Mapping):
        contact_info = visual.get("contact_sheet_info", {})
        width = _mapping(contact_info).get("width", 0)
        height = _mapping(contact_info).get("height", 0)
        byte_size = _mapping(contact_info).get("byte_size", 0)
        lines.extend(
            [
                "",
                "## Review Visual Artifacts",
                "",
                f"- status: `{visual.get('status')}`",
                f"- preview image count: `{visual.get('preview_image_count')}`",
                f"- existing preview images: `{visual.get('existing_preview_image_count')}`",
                f"- missing preview images: `{visual.get('missing_preview_image_count')}`",
                f"- contact sheet shape: `{width} x {height}`",
                f"- contact sheet bytes: `{byte_size}`",
            ]
        )
    tables = summary.get("review_table_artifacts", {})
    if isinstance(tables, Mapping):
        pass_only = _mapping(tables.get("auto_quality_pass_segments", {}))
        worklist = _mapping(tables.get("worklist", {}))
        gallery = _mapping(tables.get("gallery", {}))
        decision_template = _mapping(tables.get("decision_template", {}))
        lines.extend(
            [
                "",
                "## Review Table Artifacts",
                "",
                f"- status: `{tables.get('status')}`",
                f"- auto-quality-pass CSV status: `{pass_only.get('status')}`",
                f"- auto-quality-pass CSV row count: `{pass_only.get('row_count')}`",
                f"- worklist status: `{worklist.get('status')}`",
                f"- worklist row count: `{worklist.get('row_count')}`",
                f"- gallery status: `{gallery.get('status')}`",
                f"- gallery row count: `{gallery.get('row_count')}`",
                f"- gallery empty image src count: `{gallery.get('empty_src_count')}`",
                f"- decision template status: `{decision_template.get('status')}`",
                f"- decision template row count: `{decision_template.get('row_count')}`",
            ]
        )
    failures = summary.get("failures", [])
    if isinstance(failures, Sequence) and failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure}`")
    warnings = summary.get("warnings", [])
    if isinstance(warnings, Sequence) and warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- `{warning}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _review_next_action(status: str) -> str:
    if status == "awaiting_manual_segment_approval":
        return "review auto-quality-passed candidates and mark approved rows in segment_review_manifest.csv"
    if status == "ready_for_seed_package_build":
        return "run python -m embodied_rps.tools.build_v7_rps_seed_package"
    if status == "invalid_review_artifacts":
        return "regenerate missing or placeholder review visual artifacts before approving segments"
    return "fix invalid approval rows before building the seed package"


def _static_scissors_segments(
    *,
    run_id: str,
    run_root: Path,
    skeleton: dict[str, object],
    config: V7SegmentProposalConfig,
    segments_root: Path,
    previews_root: Path,
) -> list[dict[str, object]]:
    prompts = cast(list[str], skeleton["active_prompts"])
    ranges = _prompt_ranges(prompts, target_prompt="scissors")
    records: list[dict[str, object]] = []
    count = 0
    for start, end in ranges:
        window_start = start
        while window_start <= end:
            window_end = min(end, window_start + config.sequence_length - 1)
            if window_end - window_start + 1 >= config.min_segment_frames:
                count += 1
                records.append(
                    _write_segment_record(
                        segment_id=f"{run_id}_scissors_static_{count:06d}",
                        run_id=run_id,
                        run_root=run_root,
                        skeleton=skeleton,
                        start=window_start,
                        end=window_end,
                        proposal_role="scissors_static",
                        segments_root=segments_root,
                        previews_root=previews_root,
                        min_detection_coverage=config.min_detection_coverage,
                    )
                )
            if window_end >= end:
                break
            window_start += config.static_stride
    return records


def _transition_scissors_segments(
    *,
    run_id: str,
    run_root: Path,
    skeleton: dict[str, object],
    config: V7SegmentProposalConfig,
    segments_root: Path,
    previews_root: Path,
) -> list[dict[str, object]]:
    prompts = cast(list[str], skeleton["active_prompts"])
    records: list[dict[str, object]] = []
    for index, (prompt_start, prompt_end) in enumerate(_prompt_ranges(prompts, target_prompt="scissors"), start=1):
        end = prompt_end
        start = max(0, prompt_start - config.prefix_frames)
        if end - start + 1 > config.sequence_length:
            start = end - config.sequence_length + 1
        if end - start + 1 < config.min_segment_frames:
            continue
        records.append(
            _write_segment_record(
                segment_id=f"{run_id}_scissors_transition_{index:06d}",
                run_id=run_id,
                run_root=run_root,
                skeleton=skeleton,
                start=start,
                end=end,
                proposal_role="scissors_transition",
                segments_root=segments_root,
                previews_root=previews_root,
                min_detection_coverage=config.min_detection_coverage,
            )
        )
    return records


def _write_segment_record(
    *,
    segment_id: str,
    run_id: str,
    run_root: Path,
    skeleton: dict[str, object],
    start: int,
    end: int,
    proposal_role: str,
    segments_root: Path,
    previews_root: Path,
    min_detection_coverage: float,
) -> dict[str, object]:
    canonical = cast(NDArray[np.float32], skeleton["canonical_landmarks"])[start : end + 1].astype(np.float32, copy=True)
    detected = cast(NDArray[np.bool_], skeleton["detected"])[start : end + 1].astype(np.bool_, copy=True)
    frame_indices = cast(NDArray[np.int64], skeleton["frame_indices"])[start : end + 1].astype(np.int64, copy=True)
    times_s = cast(NDArray[np.float32], skeleton["times_s"])[start : end + 1].astype(np.float32, copy=True)
    prompts = np.asarray(cast(list[str], skeleton["active_prompts"])[start : end + 1])
    severe_jumps = _severe_jump_count(canonical[detected])
    finite = bool(np.isfinite(canonical[detected]).all()) if np.any(detected) else False
    detection_coverage = float(np.count_nonzero(detected) / max(1, detected.shape[0]))
    quality_status = (
        "auto_quality_pass"
        if finite and severe_jumps == 0 and detection_coverage >= min_detection_coverage
        else "auto_quality_failed"
    )
    segment_npz = segments_root / f"{segment_id}.npz"
    overlay_video = run_root / "scissors_pose_collection_overlay.mp4"
    preview_image = _write_segment_preview(
        overlay_video=overlay_video,
        frame_index=int(frame_indices[len(frame_indices) // 2]),
        output_path=previews_root / f"{segment_id}.jpg",
        label=segment_id,
    )
    np.savez_compressed(
        segment_npz,
        canonical_landmarks=canonical,
        detected=detected,
        frame_indices=frame_indices,
        times_s=times_s,
        active_prompts=prompts,
        metadata_json=np.asarray(json.dumps({"segment_id": segment_id, "source_run_id": run_id})),
    )
    return {
        "segment_id": segment_id,
        "source_run_id": run_id,
        "source_path": _path_from_base(run_root, base=segments_root.parent),
        "source_frame_log": _path_from_base(run_root / "scissors_pose_collection_frames.jsonl", base=segments_root.parent),
        "source_skeleton_npz": _path_from_base(run_root / "scissors_pose_collection_skeletons.npz", base=segments_root.parent),
        "source_overlay_video": _path_from_base(overlay_video, base=segments_root.parent) if overlay_video.exists() else "",
        "target_name": "scissors",
        "proposal_role": proposal_role,
        "start_frame_index": int(frame_indices[0]),
        "end_frame_index": int(frame_indices[-1]),
        "start_s": float(times_s[0]),
        "end_s": float(times_s[-1]),
        "frame_count": int(detected.shape[0]),
        "detected_frame_count": int(np.count_nonzero(detected)),
        "detection_coverage": detection_coverage,
        "severe_landmark_jump_count": severe_jumps,
        "finite": finite,
        "quality_status": quality_status,
        "review_status": "pending_manual_review",
        "approved_for_training": False,
        "skeleton_npz": _path_from_base(segment_npz, base=segments_root.parent),
        "preview_image": _path_from_base(preview_image, base=segments_root.parent) if preview_image is not None else "",
        "source_name": "v7_real_rps_seed",
    }


def _write_archived_live_segment_record(
    *,
    run_id: str,
    evidence_role: str,
    target_name: str,
    run_root: Path,
    frame_rows: Sequence[Mapping[str, object]],
    skeleton: Mapping[str, object],
    sidecar_path: Path,
    sequence_length: int,
    prefix_frames: int,
    min_segment_frames: int,
    min_detection_coverage: float,
    segments_root: Path,
    previews_root: Path,
    output_root: Path,
) -> dict[str, object]:
    canonical_all = cast(NDArray[np.float32], skeleton["canonical_landmarks"])
    detected_all = cast(NDArray[np.bool_], skeleton["detected"])
    frame_indices_all = cast(NDArray[np.int64], skeleton["frame_indices"])
    times_all = cast(NDArray[np.float32], skeleton["times_s"])
    if canonical_all.shape[0] == 0:
        raise ValueError("archived sidecar contains no frames")
    start, end = _archived_live_segment_bounds(
        frame_rows=frame_rows,
        frame_indices=frame_indices_all,
        sequence_length=sequence_length,
        prefix_frames=prefix_frames,
        min_segment_frames=min_segment_frames,
    )
    canonical = canonical_all[start : end + 1].astype(np.float32, copy=True)
    detected = detected_all[start : end + 1].astype(np.bool_, copy=True)
    frame_indices = frame_indices_all[start : end + 1].astype(np.int64, copy=True)
    times_s = times_all[start : end + 1].astype(np.float32, copy=True)
    prompts = _archived_active_prompts(frame_rows=frame_rows, frame_indices=frame_indices)
    severe_jumps = _severe_jump_count(canonical[detected])
    finite = bool(np.isfinite(canonical[detected]).all()) if np.any(detected) else False
    detection_coverage = float(np.count_nonzero(detected) / max(1, detected.shape[0]))
    quality_status = (
        "auto_quality_pass"
        if finite and severe_jumps == 0 and detection_coverage >= min_detection_coverage
        else "auto_quality_failed"
    )
    segment_id = f"{run_id}_archived_live_{target_name}_000001"
    segment_npz = segments_root / f"{segment_id}.npz"
    overlay_video = run_root / "media" / "live_camera_overlay.mp4"
    preview_image = _write_segment_preview(
        overlay_video=overlay_video,
        frame_index=int(frame_indices[len(frame_indices) // 2]),
        output_path=previews_root / f"{segment_id}.jpg",
        label=segment_id,
    )
    np.savez_compressed(
        segment_npz,
        canonical_landmarks=canonical,
        detected=detected,
        frame_indices=frame_indices,
        times_s=times_s,
        active_prompts=np.asarray(prompts),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "segment_id": segment_id,
                    "source_run_id": run_id,
                    "source_kind": "archived_live_overlay",
                    "evidence_role": evidence_role,
                    "source_skeleton_npz": _path_from_base(sidecar_path, base=output_root),
                },
                ensure_ascii=True,
            )
        ),
    )
    return {
        "segment_id": segment_id,
        "source_run_id": run_id,
        "source_path": _path_from_base(run_root, base=output_root),
        "source_frame_log": _path_from_base(run_root / "media" / "live_camera_frames.jsonl", base=output_root),
        "source_skeleton_npz": _path_from_base(sidecar_path, base=output_root),
        "source_overlay_video": _path_from_base(overlay_video, base=output_root) if overlay_video.exists() else "",
        "target_name": target_name,
        "proposal_role": f"archived_live_overlay_{evidence_role}",
        "start_frame_index": int(frame_indices[0]),
        "end_frame_index": int(frame_indices[-1]),
        "start_s": float(times_s[0]),
        "end_s": float(times_s[-1]),
        "frame_count": int(detected.shape[0]),
        "detected_frame_count": int(np.count_nonzero(detected)),
        "detection_coverage": detection_coverage,
        "severe_landmark_jump_count": severe_jumps,
        "finite": finite,
        "quality_status": quality_status,
        "review_status": "pending_manual_review",
        "approved_for_training": False,
        "skeleton_npz": _path_from_base(segment_npz, base=output_root),
        "preview_image": _path_from_base(preview_image, base=output_root) if preview_image is not None else "",
        "source_name": "v7_archived_live_overlay_seed_candidate",
        "evidence_role": evidence_role,
        "training_policy": "candidate_only_until_manual_segment_review_approval",
    }


def _write_review_manifest(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    return _write_review_manifest_preserving_decisions(path, records, _read_review_manifest_decisions(path))


def _write_review_manifest_preserving_decisions(
    path: Path,
    records: Sequence[Mapping[str, object]],
    decisions: Mapping[str, Mapping[str, str]],
) -> Path:
    fieldnames = [
        "segment_id",
        "target_name",
        "source_run_id",
        "proposal_role",
        "start_frame_index",
        "end_frame_index",
        "start_s",
        "end_s",
        "frame_count",
        "detection_coverage",
        "quality_status",
        "review_status",
        "approved_for_training",
        "review_notes",
        "skeleton_npz",
        "preview_image",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            decision = decisions.get(str(record.get("segment_id", "")), {})
            writer.writerow(
                {
                    **{key: record.get(key, "") for key in fieldnames},
                    "approved_for_training": decision.get("approved_for_training", "false"),
                    "review_status": decision.get("review_status", "pending_manual_review"),
                    "review_notes": decision.get("review_notes", ""),
                }
            )
    return path


def _read_review_manifest_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    decisions: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            segment_id = str(row.get("segment_id", ""))
            if not segment_id:
                continue
            decisions[segment_id] = {
                "approved_for_training": str(row.get("approved_for_training", "false")).strip().lower() or "false",
                "review_status": str(row.get("review_status", "pending_manual_review")).strip() or "pending_manual_review",
                "review_notes": str(row.get("review_notes", "")),
            }
    return decisions


def _write_auto_quality_pass_csv(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    fieldnames = [
        "segment_id",
        "target_name",
        "source_run_id",
        "proposal_role",
        "frame_count",
        "detection_coverage",
        "start_s",
        "end_s",
        "preview_image",
        "skeleton_npz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record.get("quality_status") != "auto_quality_pass":
                continue
            row = {field: record.get(field, "") for field in fieldnames}
            row["preview_image"] = _display_path(record.get("preview_image", ""), base=path.parent)
            row["skeleton_npz"] = _display_path(record.get("skeleton_npz", ""), base=path.parent)
            writer.writerow(row)
    return path


def _write_segment_review_packet(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    quality_counts = Counter(str(record.get("quality_status", "")) for record in records)
    role_counts = Counter(str(record.get("proposal_role", "")) for record in records)
    run_counts = Counter(str(record.get("source_run_id", "")) for record in records)
    pass_records = [record for record in records if record.get("quality_status") == "auto_quality_pass"]
    failed_records = [record for record in records if record.get("quality_status") != "auto_quality_pass"]
    lines = [
        "# V7 RPS Segment Review Packet",
        "",
        "Manual approval is required before any v7 seed NPZ, expanded dataset, or training run is generated.",
        "",
        "## Summary",
        "",
        f"- proposed segments: `{len(records)}`",
        f"- auto-quality pass: `{len(pass_records)}`",
        f"- auto-quality failed: `{len(failed_records)}`",
        f"- proposal roles: `{dict(sorted(role_counts.items()))}`",
        f"- source runs: `{dict(sorted(run_counts.items()))}`",
        f"- quality counts: `{dict(sorted(quality_counts.items()))}`",
        "",
        "## Approval Rule",
        "",
        "Edit `segment_review_manifest.csv` only after visual review. A segment is eligible for seed packaging only when:",
        "",
        "```text",
        "approved_for_training = true",
        "review_status = approved",
        "quality_status = auto_quality_pass",
        "```",
        "",
        "## Auto-Quality-Passed Candidates",
        "",
        "| Segment | Run | Role | Frames | Detection | Preview |",
        "|---|---|---|---:|---:|---|",
    ]
    for record in pass_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("segment_id", "")),
                    str(record.get("source_run_id", "")),
                    str(record.get("proposal_role", "")),
                    str(record.get("frame_count", "")),
                    f"{float(record.get('detection_coverage', 0.0)):.4f}",
                    _display_path(record.get("preview_image", ""), base=path.parent),
                ]
            )
            + " |"
        )
    if failed_records:
        lines.extend(["", "## Auto-Quality-Failed Candidates", "", "| Segment | Run | Reason | Detection | Severe jumps |", "|---|---|---|---:|---:|"])
        for record in failed_records:
            reason = "quality threshold failed"
            if float(record.get("detection_coverage", 0.0)) < 0.95:
                reason = "detection coverage below 0.95"
            if int(record.get("severe_landmark_jump_count", 0)) > 0:
                reason = f"{reason}; severe landmark jumps present"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(record.get("segment_id", "")),
                        str(record.get("source_run_id", "")),
                        reason,
                        f"{float(record.get('detection_coverage', 0.0)):.4f}",
                        str(record.get("severe_landmark_jump_count", "")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _display_path(value: object, *, base: Path) -> str:
    text = str(value)
    if not text:
        return ""
    path = Path(text)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        pass
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _path_from_base(path: Path, *, base: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(strict=False), start=base.resolve(strict=False))).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_artifact_path(value: object, *, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _seed_record_from_segment(
    record: Mapping[str, object],
    *,
    review_row: Mapping[str, object] | None = None,
    sequence_length: int,
    base: Path,
) -> dict[str, object]:
    segment_npz = _resolve_artifact_path(record["skeleton_npz"], base=base)
    with np.load(segment_npz, allow_pickle=False) as data:
        canonical = np.asarray(data["canonical_landmarks"], dtype=np.float32)
        detected = np.asarray(data["detected"], dtype=np.bool_) if "detected" in data else np.ones((canonical.shape[0],), dtype=np.bool_)
    if canonical.ndim != 3 or canonical.shape[1:] != (21, 3):
        raise ValueError(f"{segment_npz} canonical_landmarks must have shape (T,21,3)")
    selected = canonical[detected]
    if selected.shape[0] == 0:
        raise ValueError(f"{segment_npz} has no detected frames")
    padded, mask, progress, length = _resample_or_pad(selected, sequence_length=sequence_length)
    features = landmark_velocity_features(
        padded[None, ...],
        mask=mask[None, ...],
        lengths=np.asarray([length], dtype=np.int64),
    )[0]
    target_name = str(record["target_name"])
    review = review_row or {}
    return {
        "sample_id": str(record["segment_id"]),
        "label": int(TARGET_TO_LABEL[target_name]),
        "label_name": target_name,
        "target_name": target_name,
        "split": _split_for_index(0, 1),
        "length": length,
        "mask": mask,
        "progress": progress,
        "canonical_landmarks": padded,
        "features": features.astype(np.float32),
        "source_name": str(record.get("source_name") or "v7_real_rps_seed"),
        "source_path": str(record.get("source_path", "")),
        "proposal_role": str(record.get("proposal_role", "")),
        "source_run_id": str(record.get("source_run_id", "")),
        "approved_for_training": _truthy(review.get("approved_for_training", "")),
        "review_status": str(review.get("review_status", "")),
        "review_notes": str(review.get("review_notes", "")),
        "hard_example": True,
    }


def _resample_or_pad(
    canonical: NDArray[np.float32],
    *,
    sequence_length: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32], int]:
    frame_count = int(canonical.shape[0])
    if frame_count >= sequence_length:
        selected_indices = [int(round(value)) for value in np.linspace(0, frame_count - 1, sequence_length)]
        selected = canonical[selected_indices]
        length = sequence_length
    else:
        selected = canonical
        length = frame_count
    padded = np.zeros((sequence_length, 21, 3), dtype=np.float32)
    mask = np.zeros((sequence_length,), dtype=np.bool_)
    progress = np.zeros((sequence_length,), dtype=np.float32)
    padded[:length] = selected[:length]
    mask[:length] = True
    progress[:length] = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if length < sequence_length:
        padded[length:] = selected[length - 1]
        progress[length:] = 1.0
    return padded, mask, progress, length


def _write_seed_npz(path: Path, seed_records: Sequence[Mapping[str, object]], *, sequence_length: int) -> None:
    count = len(seed_records)
    np.savez_compressed(
        path,
        sample_ids=np.asarray([str(record["sample_id"]) for record in seed_records], dtype="<U96"),
        labels=np.asarray([int(record["label"]) for record in seed_records], dtype=np.int64),
        label_names=np.asarray([str(record["label_name"]) for record in seed_records], dtype="<U24"),
        target_names=np.asarray([str(record["target_name"]) for record in seed_records], dtype="<U16"),
        split_names=np.asarray([_split_for_index(index, count) for index in range(count)], dtype="<U5"),
        lengths=np.asarray([int(record["length"]) for record in seed_records], dtype=np.int64),
        mask=np.stack([cast(NDArray[np.bool_], record["mask"]) for record in seed_records]).astype(np.bool_),
        progress=np.stack([cast(NDArray[np.float32], record["progress"]) for record in seed_records]).astype(np.float32),
        canonical_landmarks=np.stack([cast(NDArray[np.float32], record["canonical_landmarks"]) for record in seed_records]).astype(np.float32),
        features=np.stack([cast(NDArray[np.float32], record["features"]) for record in seed_records]).astype(np.float32),
        source_names=np.asarray([str(record["source_name"]) for record in seed_records], dtype="<U64"),
        hard_example_flags=np.asarray([bool(record["hard_example"]) for record in seed_records], dtype=np.bool_),
        source_paths=np.asarray([str(record["source_path"]) for record in seed_records], dtype="<U512"),
        proposal_roles=np.asarray([str(record["proposal_role"]) for record in seed_records], dtype="<U64"),
        source_run_ids=np.asarray([str(record["source_run_id"]) for record in seed_records], dtype="<U64"),
        source_metadata_json=np.asarray([_seed_source_metadata_json(record) for record in seed_records]),
    )
    with np.load(path, allow_pickle=False) as data:
        validation = _validate_seed_npz_contract(data=data, count=count, sequence_length=sequence_length)
    if validation["status"] != "passed":
        raise ValueError(f"v7 seed NPZ contract validation failed: {validation['failures']}")


def _seed_source_metadata_json(record: Mapping[str, object]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"canonical_landmarks", "features", "mask", "progress"}
    }
    return json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)


def _validate_seed_npz_contract(*, data: Mapping[str, Any], count: int, sequence_length: int) -> dict[str, object]:
    keys = set(data.keys())
    failures: list[dict[str, object]] = []
    missing = [key for key in V7_SEED_NPZ_REQUIRED_KEYS if key not in keys]
    if missing:
        failures.append({"code": "missing_npz_keys", "keys": missing})
        return {"status": "failed", "required_keys": list(V7_SEED_NPZ_REQUIRED_KEYS), "failures": failures}

    expected_vector_shape = (count,)
    vector_keys = (
        "sample_ids",
        "labels",
        "label_names",
        "target_names",
        "split_names",
        "lengths",
        "source_names",
        "hard_example_flags",
        "source_paths",
        "proposal_roles",
        "source_run_ids",
        "source_metadata_json",
    )
    for key in vector_keys:
        if tuple(np.asarray(data[key]).shape) != expected_vector_shape:
            failures.append({"code": "bad_npz_vector_shape", "key": key, "shape": list(np.asarray(data[key]).shape)})

    shape_expectations = {
        "mask": (count, sequence_length),
        "progress": (count, sequence_length),
        "canonical_landmarks": (count, sequence_length, 21, 3),
        "features": (count, sequence_length, 126),
    }
    for key, expected_shape in shape_expectations.items():
        if tuple(np.asarray(data[key]).shape) != expected_shape:
            failures.append({"code": "bad_npz_array_shape", "key": key, "shape": list(np.asarray(data[key]).shape), "expected": list(expected_shape)})

    canonical = np.asarray(data["canonical_landmarks"], dtype=np.float32)
    features = np.asarray(data["features"], dtype=np.float32)
    progress = np.asarray(data["progress"], dtype=np.float32)
    mask = np.asarray(data["mask"], dtype=np.bool_)
    labels = np.asarray(data["labels"], dtype=np.int64)
    target_names = [str(value) for value in np.asarray(data["target_names"]).tolist()]
    hard_flags = np.asarray(data["hard_example_flags"], dtype=np.bool_)
    if canonical.shape == shape_expectations["canonical_landmarks"] and not np.all(np.isfinite(canonical[mask])):
        failures.append({"code": "non_finite_npz_canonical"})
    if features.shape == shape_expectations["features"] and not np.all(np.isfinite(features[mask])):
        failures.append({"code": "non_finite_npz_features"})
    if progress.shape == shape_expectations["progress"] and not np.all(np.isfinite(progress[mask])):
        failures.append({"code": "non_finite_npz_progress"})
    for index, target_name in enumerate(target_names):
        if target_name not in TARGET_TO_LABEL:
            failures.append({"code": "unsupported_npz_target", "index": index, "target_name": target_name})
        elif index < labels.shape[0] and int(labels[index]) != int(TARGET_TO_LABEL[target_name]):
            failures.append({"code": "label_target_mismatch", "index": index, "target_name": target_name, "label": int(labels[index])})
    if hard_flags.shape == expected_vector_shape and not bool(np.all(hard_flags)):
        failures.append({"code": "seed_hard_example_flags_not_all_true"})
    metadata_records = [str(value) for value in np.asarray(data["source_metadata_json"]).tolist()]
    for index, metadata_json in enumerate(metadata_records):
        try:
            loaded = json.loads(metadata_json)
        except json.JSONDecodeError:
            failures.append({"code": "invalid_source_metadata_json", "index": index})
            continue
        if not isinstance(loaded, Mapping):
            failures.append({"code": "source_metadata_not_object", "index": index})

    return {
        "status": "passed" if not failures else "failed",
        "required_keys": list(V7_SEED_NPZ_REQUIRED_KEYS),
        "failures": failures,
    }


def _write_seed_metadata(path: Path, seed_records: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in seed_records:
            handle.write(_seed_source_metadata_json(record) + "\n")


def _write_seed_quality_csv(path: Path, seed_records: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["sample_id", "target_name", "length", "source_name", "source_path", "proposal_role", "source_run_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in seed_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def _validate_seed_records(seed_records: Sequence[Mapping[str, object]], *, sequence_length: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    sample_ids = [str(record["sample_id"]) for record in seed_records]
    if len(set(sample_ids)) != len(sample_ids):
        failures.append({"code": "duplicate_sample_ids"})
    for record in seed_records:
        sample_id = str(record["sample_id"])
        target_name = str(record["target_name"])
        if target_name not in TARGET_NAMES:
            failures.append({"code": "unsupported_target", "sample_id": sample_id, "target_name": target_name})
        canonical = cast(NDArray[np.float32], record["canonical_landmarks"])
        mask = cast(NDArray[np.bool_], record["mask"])
        features = cast(NDArray[np.float32], record["features"])
        if canonical.shape != (sequence_length, 21, 3):
            failures.append({"code": "bad_canonical_shape", "sample_id": sample_id, "shape": list(canonical.shape)})
        if features.shape != (sequence_length, 126):
            failures.append({"code": "bad_features_shape", "sample_id": sample_id, "shape": list(features.shape)})
        if mask.shape != (sequence_length,):
            failures.append({"code": "bad_mask_shape", "sample_id": sample_id, "shape": list(mask.shape)})
        if not np.all(np.isfinite(canonical[mask])):
            failures.append({"code": "non_finite_canonical", "sample_id": sample_id})
        if not np.all(np.isfinite(features[mask])):
            failures.append({"code": "non_finite_features", "sample_id": sample_id})
    return {
        "status": "passed" if not failures else "failed",
        "sample_count": len(seed_records),
        "target_counts": dict(sorted(Counter(str(record["target_name"]) for record in seed_records).items())),
        "failures": failures,
    }


def _load_collection_skeleton_npz(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing skeleton NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        canonical = np.asarray(data["canonical_landmarks"], dtype=np.float32)
        detected = np.asarray(data["detected"], dtype=np.bool_)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        times_s = np.asarray(data["times_s"], dtype=np.float32)
        active_prompts = [str(value) for value in np.asarray(data["active_prompts"]).tolist()]
    if canonical.ndim != 3 or canonical.shape[1:] != (21, 3):
        raise ValueError(f"{path} canonical_landmarks must have shape (T,21,3)")
    frame_count = int(canonical.shape[0])
    if detected.shape != (frame_count,) or frame_indices.shape != (frame_count,) or times_s.shape != (frame_count,):
        raise ValueError(f"{path} sidecar arrays must share frame length")
    if len(active_prompts) != frame_count:
        raise ValueError(f"{path} active_prompts length mismatch")
    return {
        "canonical_landmarks": canonical,
        "detected": detected,
        "frame_indices": frame_indices,
        "times_s": times_s,
        "active_prompts": active_prompts,
        "frame_count": frame_count,
    }


def _select_archived_live_skeleton_sidecar(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    preferred = run_root / "media" / "live_camera_skeletons.npz"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path
        for path in run_root.rglob("*skeleton*.npz")
        if path.is_file() and "segment" not in path.name.lower()
    )
    return candidates[0] if candidates else None


def _archived_live_segment_bounds(
    *,
    frame_rows: Sequence[Mapping[str, object]],
    frame_indices: NDArray[np.int64],
    sequence_length: int,
    prefix_frames: int,
    min_segment_frames: int,
) -> tuple[int, int]:
    if frame_indices.shape[0] == 0:
        raise ValueError("frame_indices must not be empty")
    index_to_position = {int(frame_index): position for position, frame_index in enumerate(frame_indices.tolist())}
    response_positions = [
        index_to_position[int(row["frame_index"])]
        for row in frame_rows
        if bool(row.get("response_window")) and isinstance(row.get("frame_index"), (int, float)) and int(row["frame_index"]) in index_to_position
    ]
    if response_positions:
        response_start = min(response_positions)
        response_end = max(response_positions)
        start = max(0, response_start - prefix_frames)
        end = min(frame_indices.shape[0] - 1, response_end)
    else:
        end = frame_indices.shape[0] - 1
        start = max(0, end - sequence_length + 1)
    if end - start + 1 > sequence_length:
        start = end - sequence_length + 1
    if end - start + 1 < min_segment_frames and frame_indices.shape[0] >= min_segment_frames:
        start = max(0, min(start, frame_indices.shape[0] - min_segment_frames))
        end = min(frame_indices.shape[0] - 1, start + min_segment_frames - 1)
    if end - start + 1 < min_segment_frames:
        raise ValueError(f"archived segment has {end - start + 1} frames, below min_segment_frames={min_segment_frames}")
    return start, end


def _archived_active_prompts(*, frame_rows: Sequence[Mapping[str, object]], frame_indices: NDArray[np.int64]) -> list[str]:
    prompt_by_index = {
        int(row["frame_index"]): str(row.get("active_prompt") or row.get("raw_prompt") or "")
        for row in frame_rows
        if isinstance(row.get("frame_index"), (int, float))
    }
    return [prompt_by_index.get(int(frame_index), "") for frame_index in frame_indices.tolist()]


def _merge_archived_live_proposals(
    existing_records: Sequence[Mapping[str, object]],
    archive_records: Sequence[Mapping[str, object]],
    *,
    run_ids: set[str],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    new_by_id = {str(record.get("segment_id", "")): dict(record) for record in archive_records}
    for record in existing_records:
        segment_id = str(record.get("segment_id", ""))
        source_run_id = str(record.get("source_run_id", ""))
        proposal_role = str(record.get("proposal_role", ""))
        if segment_id in new_by_id:
            continue
        if source_run_id in run_ids and proposal_role.startswith("archived_live_overlay_"):
            continue
        merged.append(dict(record))
    merged.extend(new_by_id[segment_id] for segment_id in sorted(new_by_id))
    return sorted(merged, key=lambda record: (str(record.get("source_run_id", "")), str(record.get("segment_id", ""))))


def _write_archived_live_overlay_segment_summary_md(
    path: Path,
    *,
    summary: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> Path:
    lines = [
        "# V7 Archived Live Overlay Segment Summary",
        "",
        "This file records review-gated archived live overlay candidates. It does not approve seeds, build the v7 seed NPZ, train, validate, or promote v7.",
        "",
        "## Summary",
        "",
        f"- status: `{summary.get('status')}`",
        f"- proposed archived segments: `{summary.get('proposed_segment_count')}`",
        f"- auto-quality pass: `{summary.get('auto_quality_pass_count')}`",
        f"- auto-quality failed: `{summary.get('auto_quality_failed_count')}`",
        f"- approved segments: `{summary.get('approved_segment_count')}`",
        "",
        "## Candidates",
        "",
        "| Segment | Target | Role | Frames | Detection | Quality | Preview |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("segment_id", "")),
                    str(record.get("target_name", "")),
                    str(record.get("proposal_role", "")),
                    str(record.get("frame_count", "")),
                    f"{float(record.get('detection_coverage', 0.0)):.4f}",
                    str(record.get("quality_status", "")),
                    str(record.get("preview_image", "")),
                ]
            )
            + " |"
        )
    skipped = summary.get("skipped_runs", [])
    if isinstance(skipped, Sequence) and skipped:
        lines.extend(["", "## Skipped Runs", "", "| Run | Status | Reason |", "|---|---|---|"])
        for item in skipped:
            row = _mapping(item)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("run_id", "")),
                        str(row.get("status", "")),
                        str(row.get("reason", row.get("next_action", ""))),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _relative_extraction_summary(summary: Mapping[str, object], *, project_root: Path) -> dict[str, object]:
    clean = dict(summary)
    for key in ("skeleton_npz", "overlay_video", "frame_log"):
        value = clean.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        clean[key] = _path_from_base(path, base=project_root) if path.is_absolute() else path.as_posix()
    return clean


def _extract_archived_live_overlay_sidecar(
    *,
    run_root: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, object]:
    overlay_video = run_root / "media" / "live_camera_overlay.mp4"
    frame_log = run_root / "media" / "live_camera_frames.jsonl"
    if output_path.exists() and not overwrite:
        return {"status": "passed", "skeleton_npz": output_path.as_posix(), "reason": "existing_sidecar"}
    if not overlay_video.exists():
        return {"status": "missing_overlay_video", "overlay_video": overlay_video.as_posix()}
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"status": "missing_mediapipe_dependency", "reason": str(exc)}
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        return {"status": "missing_mediapipe_hands"}

    from embodied_rps.tools.run_realtime_skeleton_predictor import canonicalize_mediapipe_landmarks

    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        return {"status": "could_not_open_overlay_video", "overlay_video": overlay_video.as_posix()}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    frame_rows = _read_jsonl_if_exists(frame_log)
    prompt_by_position = [str(row.get("active_prompt") or row.get("raw_prompt") or "") for row in frame_rows]
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    canonical_frames: list[NDArray[np.float32]] = []
    detected: list[bool] = []
    frame_indices: list[int] = []
    times_s: list[float] = []
    prompts: list[str] = []
    last_valid = np.zeros((21, 3), dtype=np.float32)
    try:
        frame_index = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            hand_landmarks = list(result.multi_hand_landmarks or [])
            if hand_landmarks:
                raw = np.asarray(
                    [[float(point.x), float(point.y), float(point.z)] for point in hand_landmarks[0].landmark],
                    dtype=np.float32,
                )
                last_valid = canonicalize_mediapipe_landmarks(raw)
                detected.append(True)
            else:
                detected.append(False)
            canonical_frames.append(last_valid.astype(np.float32, copy=True))
            frame_indices.append(frame_index + 1)
            times_s.append(frame_index / fps)
            prompts.append(prompt_by_position[frame_index] if frame_index < len(prompt_by_position) else "")
            frame_index += 1
    finally:
        hands.close()
        capture.release()
    landmarks = np.stack(canonical_frames, axis=0).astype(np.float32) if canonical_frames else np.zeros((0, 21, 3), dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": overlay_video.as_posix(),
        "frame_log": frame_log.as_posix() if frame_log.exists() else None,
        "contract": "archived live overlay MediaPipe skeleton sidecar for v7 manual review candidates",
    }
    np.savez_compressed(
        output_path,
        canonical_landmarks=landmarks,
        detected=np.asarray(detected, dtype=np.bool_),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        times_s=np.asarray(times_s, dtype=np.float32),
        active_prompts=np.asarray(prompts),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )
    detected_count = int(np.count_nonzero(np.asarray(detected, dtype=np.bool_)))
    return {
        "status": "passed",
        "skeleton_npz": output_path.as_posix(),
        "frame_count": len(frame_indices),
        "detected_frame_count": detected_count,
        "detection_rate": float(detected_count / max(1, len(frame_indices))),
    }


def _prompt_ranges(prompts: Sequence[str], *, target_prompt: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, prompt in enumerate(prompts):
        if prompt == target_prompt and start is None:
            start = index
        if start is not None and (prompt != target_prompt or index == len(prompts) - 1):
            end = index if prompt == target_prompt and index == len(prompts) - 1 else index - 1
            ranges.append((start, end))
            start = None
    return ranges


def _severe_jump_count(canonical: NDArray[np.float32]) -> int:
    if canonical.shape[0] < 3:
        return 0
    deltas = np.linalg.norm(np.diff(canonical, axis=0), axis=2).mean(axis=1)
    median = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - median)))
    threshold = max(1.5, median + 12.0 * mad)
    return int(np.count_nonzero(deltas > threshold))


def _split_for_index(index: int, count: int) -> str:
    train_cutoff = int(round(count * 0.70))
    val_cutoff = int(round(count * 0.85))
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def _discover_heldout_test_roots(dataset_search_root: Path) -> list[Path]:
    if not dataset_search_root.exists():
        return []
    roots: list[Path] = []
    for path in sorted(dataset_search_root.rglob("test"), key=lambda value: value.as_posix()):
        if path.is_dir() and list(path.rglob("*.mp4")):
            roots.append(path)
    return roots


def _reject_heldout_path(source_path: str, context: Path) -> None:
    if _is_heldout_path(source_path):
        raise ValueError(f"{context} contains held-out test source path: {source_path}")


def _is_heldout_path(source_path: str) -> bool:
    normalized = source_path.replace("\\", "/").lower()
    return "/test/" in normalized or normalized.endswith("/test")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path) if path.exists() else []


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, dict) else {}


def _most_common(values: Iterable[object]) -> str | None:
    counter = Counter(str(value) for value in values if value is not None and str(value) != "")
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _under_project(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _write_placeholder_contact_sheet(path: Path) -> Path:
    # 1x1 transparent PNG; real video thumbnails are not required for non-video unit tests.
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    path.write_bytes(png_bytes)
    return path


def _write_segment_preview(*, overlay_video: Path, frame_index: int, output_path: Path, label: str) -> Path | None:
    if not overlay_video.exists():
        return None
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(overlay_video))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.putText(frame, label[:48], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    output_path.write_bytes(encoded.tobytes())
    return output_path if output_path.exists() else None


def _resolve_review_preview_paths(*, base: Path, records: Sequence[Mapping[str, object]]) -> list[Path]:
    preview_paths: list[Path] = []
    for record in records:
        raw_preview = str(record.get("preview_image", "")).strip()
        if not raw_preview:
            continue
        preview_path = Path(raw_preview)
        if not preview_path.is_absolute():
            preview_path = base / preview_path
        if preview_path.exists():
            preview_paths.append(preview_path)
    return preview_paths


def _write_review_contact_sheet(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    preview_paths = _resolve_review_preview_paths(base=path.parent, records=records)
    if not preview_paths:
        return _write_placeholder_contact_sheet(path)
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return _write_placeholder_contact_sheet(path)
    thumbs = []
    for preview_path in preview_paths:
        encoded = np.frombuffer(preview_path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        thumb_h = 140
        thumb_w = max(1, int(round(width * thumb_h / max(1, height))))
        thumbs.append(cv2.resize(image, (thumb_w, thumb_h)))
    if not thumbs:
        return _write_placeholder_contact_sheet(path)
    rows = []
    for start in range(0, len(thumbs), 4):
        row = thumbs[start : start + 4]
        max_h = max(thumb.shape[0] for thumb in row)
        padded = [
            cv2.copyMakeBorder(thumb, 0, max_h - thumb.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for thumb in row
        ]
        rows.append(np.hstack(padded))
    max_w = max(row.shape[1] for row in rows)
    padded_rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, max_w - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.vstack(padded_rows))
    if not ok:
        return _write_placeholder_contact_sheet(path)
    path.write_bytes(encoded.tobytes())
    return path


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "V7ArchivedLiveOverlayProposalConfig",
    "V7SeedManifestConfig",
    "V7SegmentProposalConfig",
    "apply_v7_segment_review_decisions",
    "audit_v7_segment_review_readiness",
    "build_v7_rps_seed_package",
    "propose_v7_archived_live_overlay_segments",
    "propose_v7_rps_segments",
    "write_v7_archived_live_candidate_manifest",
    "write_v7_segment_review_coverage_report",
    "write_v7_segment_review_decision_template",
    "write_v7_seed_manifest",
    "write_v7_segment_review_worklist",
]
