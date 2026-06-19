"""Audit and propose v7d prompt-pose collection seed candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_rps.v7_rps_seed_package import (
    write_v7_segment_review_decision_template,
    write_v7_segment_review_worklist,
)

BRANCH_LABEL = "v7d_real_seeded_two_stage_prompt_window_guard"
DEFAULT_COLLECTION_ROOT = Path("artifacts/realtime_scissors_pose_collection_20260617")
DEFAULT_AUDIT_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_audit_20260618")
DEFAULT_REVIEW_ROOT = Path("artifacts/real_skeleton_v7d_prompt_pose_collection_review_20260618")
DEFAULT_RUN_RELS: tuple[Path, ...] = (
    Path("rock/run_20260618_182913"),
    Path("paper/run_20260618_183548"),
    Path("scissors/run_20260618_183700"),
)
TARGET_NAMES: tuple[str, ...] = ("rock", "paper", "scissors")
ROLE_BY_TARGET: dict[str, str] = {
    "rock": "rock_wait_prompt_window",
    "paper": "hard_paper_prompt_window",
    "scissors": "scissors_boundary_control",
}
SOURCE_NAME_BY_TARGET: dict[str, str] = {
    "rock": "v7d_prompt_pose_rock_wait_hard_negative",
    "paper": "v7d_prompt_pose_hard_paper_rescue",
    "scissors": "v7d_prompt_pose_scissors_boundary_control",
}
PATH_FIELDS_TO_AUDIT: tuple[str, ...] = (
    "source_path",
    "source_frame_log",
    "source_skeleton_npz",
    "source_overlay_video",
    "skeleton_npz",
    "preview_image",
)


@dataclass(frozen=True)
class V7DPromptPoseCollectionConfig:
    """Inputs for auditing the completed prompt-pose collection runs."""

    project_root: Path = field(default_factory=Path.cwd)
    collection_root: Path = DEFAULT_COLLECTION_ROOT
    run_rels: tuple[Path, ...] = DEFAULT_RUN_RELS
    output_root: Path = DEFAULT_AUDIT_ROOT
    target_prompt: str = "scissors"
    min_detection_rate: float = 0.95
    min_prompt_windows: int = 10


@dataclass(frozen=True)
class V7DPromptPoseSegmentProposalConfig:
    """Inputs for writing review-gated v7d prompt-pose segment proposals."""

    project_root: Path = field(default_factory=Path.cwd)
    collection_root: Path = DEFAULT_COLLECTION_ROOT
    run_rels: tuple[Path, ...] = DEFAULT_RUN_RELS
    output_root: Path = DEFAULT_REVIEW_ROOT
    target_prompt: str = "scissors"
    sequence_length: int = 72
    prefix_frames: int = 24
    min_segment_frames: int = 30
    min_detection_coverage: float = 0.95


def audit_v7d_prompt_pose_collections(config: V7DPromptPoseCollectionConfig) -> dict[str, object]:
    """Audit collection-level readiness without proposing or approving segments."""

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    for run_rel in config.run_rels:
        run_root = _resolve_path(project_root, config.collection_root / run_rel)
        descriptor = _load_run_descriptor(project_root=project_root, run_root=run_root)
        prompt_ranges = _prompt_ranges(descriptor.active_prompts, target_prompt=config.target_prompt)
        finite = bool(np.isfinite(descriptor.canonical).all())
        prompt_count = len(prompt_ranges)
        source_response_prompt = _most_common(str(row.get("response_prompt", "")) for row in descriptor.frame_rows)
        pre_fix_mismatch = source_response_prompt != config.target_prompt
        failures: list[str] = []
        if descriptor.summary_status != "passed":
            failures.append("summary_status_not_passed")
        if descriptor.detection_rate < config.min_detection_rate:
            failures.append("detection_rate_below_threshold")
        if not finite:
            failures.append("non_finite_canonical_landmarks")
        if len(descriptor.frame_rows) != descriptor.frame_count:
            failures.append("frame_log_npz_length_mismatch")
        if prompt_count < config.min_prompt_windows:
            failures.append("insufficient_target_prompt_windows")
        for value in descriptor.path_values():
            if _is_heldout_test_path(value):
                failures.append("heldout_test_path_present")
        run_rows.append(
            {
                "collection_label": descriptor.collection_label,
                "run_id": descriptor.run_id,
                "run_root": _display_path(descriptor.run_root, base=project_root),
                "summary_status": descriptor.summary_status,
                "frame_count": descriptor.frame_count,
                "frame_log_count": len(descriptor.frame_rows),
                "detected_frame_count": descriptor.detected_frame_count,
                "detection_rate": descriptor.detection_rate,
                "finite_canonical_landmarks": finite,
                "target_prompt": config.target_prompt,
                "target_prompt_window_count": prompt_count,
                "source_response_prompt": source_response_prompt,
                "effective_response_prompt": config.target_prompt,
                "pre_fix_response_prompt_mismatch": pre_fix_mismatch,
                "quality_status": "audit_passed" if not failures else "audit_failed",
                "failures": failures,
            }
        )

    status = "passed" if all(row["quality_status"] == "audit_passed" for row in run_rows) else "failed"
    summary: dict[str, object] = {
        "status": status,
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "collection_root": _display_path(_resolve_path(project_root, config.collection_root), base=project_root),
        "target_prompt": config.target_prompt,
        "run_count": len(run_rows),
        "collection_counts": dict(sorted(Counter(str(row["collection_label"]) for row in run_rows).items())),
        "target_prompt_window_counts": {
            str(row["collection_label"]): int(row["target_prompt_window_count"]) for row in run_rows
        },
        "pre_fix_response_prompt_mismatch_runs": [
            row["run_id"] for row in run_rows if bool(row["pre_fix_response_prompt_mismatch"])
        ],
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7d candidate metadata",
        "prompt_window_policy": (
            "labels come from the chosen collection mode; proposed seed windows are prompt-conditioned temporal "
            "segments from active_prompt == scissors, not independent preview thumbnails"
        ),
        "run_rows": run_rows,
    }
    _write_audit_outputs(output_root=output_root, summary=summary, rows=run_rows)
    return summary


def propose_v7d_prompt_pose_segments(config: V7DPromptPoseSegmentProposalConfig) -> dict[str, object]:
    """Write review-gated prompt-pose candidate segments without approving them."""

    if config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if config.prefix_frames < 0:
        raise ValueError("prefix_frames must be non-negative")
    if config.min_segment_frames <= 0:
        raise ValueError("min_segment_frames must be positive")

    project_root = config.project_root.resolve()
    output_root = _resolve_path(project_root, config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    segments_root = output_root / "segments"
    previews_root = output_root / "previews"
    segments_root.mkdir(parents=True, exist_ok=True)
    previews_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    skipped_runs: list[dict[str, object]] = []
    for run_rel in config.run_rels:
        run_root = _resolve_path(project_root, config.collection_root / run_rel)
        descriptor = _load_run_descriptor(project_root=project_root, run_root=run_root)
        if descriptor.collection_label not in TARGET_NAMES:
            skipped_runs.append(
                {
                    "run_root": _display_path(run_root, base=project_root),
                    "status": "invalid_collection_label",
                    "collection_label": descriptor.collection_label,
                }
            )
            continue
        if len(descriptor.frame_rows) != descriptor.frame_count:
            raise ValueError(f"{run_root} frame log and skeleton NPZ length mismatch")
        for window_index, (prompt_start, prompt_end) in enumerate(
            _prompt_ranges(descriptor.active_prompts, target_prompt=config.target_prompt),
            start=1,
        ):
            start = max(0, prompt_start - config.prefix_frames)
            end = prompt_end
            if end - start + 1 < config.min_segment_frames:
                continue
            records.append(
                _write_segment_record(
                    descriptor=descriptor,
                    project_root=project_root,
                    output_root=output_root,
                    segments_root=segments_root,
                    previews_root=previews_root,
                    target_prompt=config.target_prompt,
                    window_index=window_index,
                    prompt_start=prompt_start,
                    prompt_end=prompt_end,
                    start=start,
                    end=end,
                    min_detection_coverage=config.min_detection_coverage,
                )
            )

    records = sorted(records, key=lambda row: (str(row["target_name"]), str(row["source_run_id"]), str(row["segment_id"])))
    proposed_path = output_root / "proposed_segments.jsonl"
    with proposed_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
    review_manifest = _write_review_manifest(output_root / "segment_review_manifest.csv", records)
    pass_csv = _write_auto_quality_pass_csv(output_root / "auto_quality_pass_segments.csv", records)
    packet = _write_review_packet(output_root / "segment_review_packet.md", records=records, skipped_runs=skipped_runs)
    contact_sheet = _write_review_contact_sheet(output_root / "review_contact_sheet.png", records=records, base=output_root)
    worklist = _relativize_paths(write_v7_segment_review_worklist(output_root=output_root), project_root=project_root)
    decision_template = _relativize_paths(
        write_v7_segment_review_decision_template(output_root=output_root),
        project_root=project_root,
    )
    (output_root / "segment_review_worklist_summary.json").write_text(
        json.dumps(_json_ready(worklist), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "segment_review_decision_template_summary.json").write_text(
        json.dumps(_json_ready(decision_template), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quality_counts = Counter(str(record.get("quality_status", "")) for record in records)
    target_counts = Counter(str(record.get("target_name", "")) for record in records)
    role_counts = Counter(str(record.get("proposal_role", "")) for record in records)
    summary: dict[str, object] = {
        "status": "awaiting_manual_review" if records else "no_segments_proposed",
        "branch_label": BRANCH_LABEL,
        "output_root": _display_path(output_root, base=project_root),
        "proposed_segment_count": len(records),
        "auto_quality_pass_count": int(quality_counts.get("auto_quality_pass", 0)),
        "auto_quality_failed_count": int(quality_counts.get("auto_quality_failed", 0)),
        "target_counts": dict(sorted(target_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "skipped_runs": skipped_runs,
        "proposed_segments": _display_path(proposed_path, base=project_root),
        "segment_review_manifest": _display_path(review_manifest, base=project_root),
        "auto_quality_pass_segments": _display_path(pass_csv, base=project_root),
        "segment_review_packet": _display_path(packet, base=project_root),
        "review_contact_sheet": _display_path(contact_sheet, base=project_root),
        "segment_review_worklist": worklist,
        "segment_review_decision_template": decision_template,
        "review_gate": "manual approval required before any v7d seed NPZ, dataset generation, training, or promotion",
        "heldout_policy": "heldout */test MP4s remain validation-only and are rejected from v7d candidate metadata",
        "prompt_window_policy": "candidate windows are selected from active_prompt == scissors and keep prefix context",
    }
    (output_root / "segment_proposal_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "segment_proposal_summary.md").write_text(_proposal_summary_markdown(summary), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class _RunDescriptor:
    project_root: Path
    run_root: Path
    run_id: str
    collection_label: str
    summary_status: str
    frame_log: Path
    skeleton_npz: Path
    overlay_video: Path | None
    frame_rows: tuple[dict[str, object], ...]
    canonical: NDArray[np.float32]
    detected: NDArray[np.bool_]
    frame_indices: NDArray[np.int64]
    times_s: NDArray[np.float32]
    active_prompts: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return int(self.canonical.shape[0])

    @property
    def detected_frame_count(self) -> int:
        return int(np.count_nonzero(self.detected))

    @property
    def detection_rate(self) -> float:
        return float(self.detected_frame_count / max(1, self.frame_count))

    def path_values(self) -> list[str]:
        values = [self.run_root.as_posix(), self.frame_log.as_posix(), self.skeleton_npz.as_posix()]
        if self.overlay_video is not None:
            values.append(self.overlay_video.as_posix())
        return values


def _load_run_descriptor(*, project_root: Path, run_root: Path) -> _RunDescriptor:
    if not run_root.exists():
        raise FileNotFoundError(f"Missing prompt-pose collection run: {run_root}")
    summary = _read_json_object(run_root / "summary" / "quality_summary.json")
    collection_label = str(summary.get("collection_label") or run_root.parent.name).strip().lower()
    if collection_label not in TARGET_NAMES:
        collection_label = _label_from_collection_files(run_root)
    frame_log = _single_glob(run_root, "*_pose_collection_frames.jsonl")
    skeleton_npz = _single_glob(run_root, "*_pose_collection_skeletons.npz")
    overlay_candidates = sorted(run_root.glob("*_pose_collection_overlay.mp4"))
    frame_rows = tuple(_read_jsonl(frame_log))
    with np.load(skeleton_npz, allow_pickle=False) as data:
        canonical = np.asarray(data["canonical_landmarks"], dtype=np.float32)
        detected = np.asarray(data["detected"], dtype=np.bool_)
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        times_s = np.asarray(data["times_s"], dtype=np.float32)
        active_prompts = tuple(str(value) for value in np.asarray(data["active_prompts"]).tolist())
    if canonical.ndim != 3 or canonical.shape[1:] != (21, 3):
        raise ValueError(f"{skeleton_npz} canonical_landmarks must have shape (T,21,3)")
    frame_count = int(canonical.shape[0])
    if detected.shape != (frame_count,) or frame_indices.shape != (frame_count,) or times_s.shape != (frame_count,):
        raise ValueError(f"{skeleton_npz} sidecar arrays must share frame length")
    if len(active_prompts) != frame_count:
        raise ValueError(f"{skeleton_npz} active_prompts length mismatch")
    return _RunDescriptor(
        project_root=project_root,
        run_root=run_root,
        run_id=run_root.name,
        collection_label=collection_label,
        summary_status=str(summary.get("status", "")).strip(),
        frame_log=frame_log,
        skeleton_npz=skeleton_npz,
        overlay_video=overlay_candidates[0] if overlay_candidates else None,
        frame_rows=frame_rows,
        canonical=canonical,
        detected=detected,
        frame_indices=frame_indices,
        times_s=times_s,
        active_prompts=active_prompts,
    )


def _write_segment_record(
    *,
    descriptor: _RunDescriptor,
    project_root: Path,
    output_root: Path,
    segments_root: Path,
    previews_root: Path,
    target_prompt: str,
    window_index: int,
    prompt_start: int,
    prompt_end: int,
    start: int,
    end: int,
    min_detection_coverage: float,
) -> dict[str, object]:
    segment_id = f"v7d_{descriptor.collection_label}_{descriptor.run_id}_prompt_scissors_{window_index:03d}"
    segment_npz = segments_root / f"{segment_id}.npz"
    preview = previews_root / f"{segment_id}.png"
    canonical = descriptor.canonical[start : end + 1].astype(np.float32, copy=True)
    detected = descriptor.detected[start : end + 1].astype(np.bool_, copy=True)
    frame_indices = descriptor.frame_indices[start : end + 1].astype(np.int64, copy=True)
    times_s = descriptor.times_s[start : end + 1].astype(np.float32, copy=True)
    prompts = np.asarray(descriptor.active_prompts[start : end + 1])
    detection_coverage = float(np.count_nonzero(detected) / max(1, detected.shape[0]))
    severe_jumps = _severe_jump_count(canonical)
    source_response_prompt = _most_common(str(row.get("response_prompt", "")) for row in descriptor.frame_rows[start : end + 1])
    pre_fix_mismatch = source_response_prompt != target_prompt
    metadata = {
        "branch_label": BRANCH_LABEL,
        "segment_id": segment_id,
        "collection_label": descriptor.collection_label,
        "target_name": descriptor.collection_label,
        "source_run_id": descriptor.run_id,
        "prompt_conditioned_sequence": True,
        "target_prompt": target_prompt,
        "source_response_prompt": source_response_prompt,
        "effective_response_prompt": target_prompt,
        "pre_fix_response_prompt_mismatch": pre_fix_mismatch,
        "prompt_start_frame_index": int(descriptor.frame_indices[prompt_start]),
        "prompt_end_frame_index": int(descriptor.frame_indices[prompt_end]),
        "prefix_frames": int(prompt_start - start),
        "training_policy": "candidate_only_until_manual_segment_review_approval",
        "heldout_policy": "nonheldout_collection_candidate_validation_only_test_paths_rejected",
    }
    np.savez_compressed(
        segment_npz,
        canonical_landmarks=canonical,
        detected=detected,
        frame_indices=frame_indices,
        times_s=times_s,
        active_prompts=prompts,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=True)),
    )
    if descriptor.overlay_video is None or _write_segment_preview(
        overlay_video=descriptor.overlay_video,
        frame_index=int(descriptor.frame_indices[prompt_start]),
        output_path=preview,
        label=f"{descriptor.collection_label} {window_index:03d}",
    ) is None:
        _write_placeholder_png(preview)
    quality_status = (
        "auto_quality_pass"
        if np.isfinite(canonical).all() and detection_coverage >= min_detection_coverage and severe_jumps == 0
        else "auto_quality_failed"
    )
    record = {
        "segment_id": segment_id,
        "target_name": descriptor.collection_label,
        "source_name": SOURCE_NAME_BY_TARGET[descriptor.collection_label],
        "source_run_id": descriptor.run_id,
        "proposal_role": ROLE_BY_TARGET[descriptor.collection_label],
        "v7d_seed_role": ROLE_BY_TARGET[descriptor.collection_label],
        "start_frame_index": int(frame_indices[0]),
        "end_frame_index": int(frame_indices[-1]),
        "prompt_start_frame_index": int(descriptor.frame_indices[prompt_start]),
        "prompt_end_frame_index": int(descriptor.frame_indices[prompt_end]),
        "start_s": float(times_s[0]) if times_s.shape[0] else 0.0,
        "end_s": float(times_s[-1]) if times_s.shape[0] else 0.0,
        "frame_count": int(canonical.shape[0]),
        "prompt_window_frame_count": int(prompt_end - prompt_start + 1),
        "prefix_frame_count": int(prompt_start - start),
        "detection_coverage": detection_coverage,
        "severe_landmark_jump_count": severe_jumps,
        "quality_status": quality_status,
        "review_status": "pending_manual_review",
        "approved_for_training": False,
        "review_notes": "",
        "skeleton_npz": _display_path(segment_npz, base=output_root),
        "preview_image": _display_path(preview, base=output_root),
        "source_path": _display_path(descriptor.run_root, base=project_root),
        "source_frame_log": _display_path(descriptor.frame_log, base=project_root),
        "source_skeleton_npz": _display_path(descriptor.skeleton_npz, base=project_root),
        "source_overlay_video": _display_path(descriptor.overlay_video, base=project_root)
        if descriptor.overlay_video is not None
        else "",
        "collection_label": descriptor.collection_label,
        "prompt_conditioned_sequence": True,
        "target_prompt": target_prompt,
        "source_response_prompt": source_response_prompt,
        "effective_response_prompt": target_prompt,
        "pre_fix_response_prompt_mismatch": pre_fix_mismatch,
        "training_policy": "candidate_only_until_manual_segment_review_approval",
        "heldout_policy": "nonheldout_collection_candidate_validation_only_test_paths_rejected",
    }
    _reject_heldout_metadata(record, context=descriptor.run_root)
    return record


def _write_audit_outputs(*, output_root: Path, summary: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    (output_root / "collection_audit_summary.json").write_text(
        json.dumps(_json_ready(dict(summary)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        output_root / "collection_audit_runs.csv",
        (
            "collection_label",
            "run_id",
            "run_root",
            "summary_status",
            "frame_count",
            "frame_log_count",
            "detected_frame_count",
            "detection_rate",
            "finite_canonical_landmarks",
            "target_prompt_window_count",
            "source_response_prompt",
            "effective_response_prompt",
            "pre_fix_response_prompt_mismatch",
            "quality_status",
        ),
        rows,
    )
    (output_root / "collection_audit_summary.md").write_text(_audit_markdown(summary), encoding="utf-8")


def _write_review_manifest(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    fieldnames = (
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
    )
    _write_csv(path, fieldnames, records, override={"approved_for_training": "false", "review_status": "pending_manual_review"})
    return path


def _write_auto_quality_pass_csv(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    fieldnames = (
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
    )
    passed = [record for record in records if record.get("quality_status") == "auto_quality_pass"]
    _write_csv(path, fieldnames, passed)
    return path


def _write_review_packet(path: Path, *, records: Sequence[Mapping[str, object]], skipped_runs: Sequence[Mapping[str, object]]) -> Path:
    role_counts = Counter(str(record.get("proposal_role", "")) for record in records)
    quality_counts = Counter(str(record.get("quality_status", "")) for record in records)
    lines = [
        "# V7d Prompt-Pose Segment Review Packet",
        "",
        "Manual approval is required before seed packaging, dataset generation, training, or promotion.",
        "",
        "## Summary",
        "",
        f"- proposed segments: `{len(records)}`",
        f"- quality counts: `{dict(sorted(quality_counts.items()))}`",
        f"- role counts: `{dict(sorted(role_counts.items()))}`",
        "- selection policy: `active_prompt == scissors` with prefix context",
        "- label policy: collection mode provides the target label; model decisions are not labels",
        "",
        "## Auto-Quality-Passed Candidates",
        "",
        "| Segment | Target | Role | Frames | Detection | Pre-fix mismatch | Preview |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for record in records:
        if record.get("quality_status") != "auto_quality_pass":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("segment_id", "")),
                    str(record.get("target_name", "")),
                    str(record.get("proposal_role", "")),
                    str(record.get("frame_count", "")),
                    f"{float(record.get('detection_coverage', 0.0)):.4f}",
                    str(record.get("pre_fix_response_prompt_mismatch", "")),
                    str(record.get("preview_image", "")),
                ]
            )
            + " |"
        )
    if skipped_runs:
        lines.extend(["", "## Skipped Runs", "", "| Run | Status |", "|---|---|"])
        for row in skipped_runs:
            lines.append(f"| {row.get('run_root', '')} | {row.get('status', '')} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _audit_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V7d Prompt-Pose Collection Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Branch: `{summary.get('branch_label')}`",
        f"- Target prompt: `{summary.get('target_prompt')}`",
        f"- Collection counts: `{summary.get('collection_counts')}`",
        f"- Target prompt windows: `{summary.get('target_prompt_window_counts')}`",
        f"- Pre-fix response-prompt mismatch runs: `{summary.get('pre_fix_response_prompt_mismatch_runs')}`",
        "",
        "The rock run is usable when marked as a pre-fix response-prompt mismatch because v7d crops from the bounded `PROMPT SCISSORS` window.",
        "",
    ]
    return "\n".join(lines)


def _proposal_summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# V7d Prompt-Pose Segment Proposal Summary",
            "",
            f"- Status: `{summary.get('status')}`",
            f"- Branch: `{summary.get('branch_label')}`",
            f"- Proposed segments: `{summary.get('proposed_segment_count')}`",
            f"- Auto-quality pass: `{summary.get('auto_quality_pass_count')}`",
            f"- Auto-quality failed: `{summary.get('auto_quality_failed_count')}`",
            f"- Target counts: `{summary.get('target_counts')}`",
            f"- Role counts: `{summary.get('role_counts')}`",
            "- Review gate: manual approval required before seed packaging, dataset generation, training, or promotion.",
            "",
        ]
    )


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    override: Mapping[str, object] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            values = {field: row.get(field, "") for field in fieldnames}
            if override:
                values.update(override)
            writer.writerow(values)


def _label_from_collection_files(run_root: Path) -> str:
    for path in run_root.glob("*_pose_collection_frames.jsonl"):
        label = path.name.split("_pose_collection_frames.jsonl", 1)[0].lower()
        if label in TARGET_NAMES:
            return label
    return ""


def _single_glob(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} in {root}, found {len(matches)}")
    return matches[0]


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


def _write_placeholder_png(path: Path) -> Path:
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index) - 1))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.putText(frame, label[:48], (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return None
    output_path.write_bytes(encoded.tobytes())
    return output_path


def _write_review_contact_sheet(path: Path, *, records: Sequence[Mapping[str, object]], base: Path) -> Path:
    preview_paths: list[Path] = []
    for record in records:
        if record.get("quality_status") != "auto_quality_pass":
            continue
        raw = str(record.get("preview_image", "")).strip()
        if not raw:
            continue
        preview_path = Path(raw)
        if not preview_path.is_absolute():
            preview_path = base / preview_path
        if preview_path.exists():
            preview_paths.append(preview_path)
    if not preview_paths:
        return _write_placeholder_png(path)
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return _write_placeholder_png(path)
    thumbs = []
    for preview_path in preview_paths:
        encoded = np.frombuffer(preview_path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        thumb_h = 120
        thumb_w = max(1, int(round(width * thumb_h / max(1, height))))
        thumbs.append(cv2.resize(image, (thumb_w, thumb_h)))
    if not thumbs:
        return _write_placeholder_png(path)
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
    ok, encoded = cv2.imencode(".png", np.vstack(padded_rows))
    if not ok:
        return _write_placeholder_png(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())
    return path


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
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


def _reject_heldout_metadata(row: Mapping[str, object], *, context: Path) -> None:
    for field_name in PATH_FIELDS_TO_AUDIT:
        value = str(row.get(field_name, "")).strip()
        if value and _is_heldout_test_path(value):
            raise ValueError(f"{context} contains held-out test candidate path in {field_name}: {value}")


def _is_heldout_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower().rstrip("/")
    return "/test/" in normalized or normalized.endswith("/test")


def _most_common(values: Sequence[object] | Any) -> str:
    counter = Counter(str(value).strip() for value in values if str(value).strip())
    return counter.most_common(1)[0][0] if counter else ""


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _display_path(path: object, *, base: Path) -> str:
    if not path:
        return ""
    resolved = Path(str(path))
    try:
        return resolved.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_ready(value: Any) -> Any:
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


def _relativize_paths(value: Any, *, project_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _relativize_paths(item, project_root=project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_paths(item, project_root=project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_relativize_paths(item, project_root=project_root) for item in value)
    if isinstance(value, str) and value:
        path = Path(value)
        if path.is_absolute():
            return _display_path(path, base=project_root)
    return value


__all__ = [
    "BRANCH_LABEL",
    "DEFAULT_AUDIT_ROOT",
    "DEFAULT_COLLECTION_ROOT",
    "DEFAULT_REVIEW_ROOT",
    "DEFAULT_RUN_RELS",
    "V7DPromptPoseCollectionConfig",
    "V7DPromptPoseSegmentProposalConfig",
    "audit_v7d_prompt_pose_collections",
    "propose_v7d_prompt_pose_segments",
]
