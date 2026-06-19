"""V4 calibration-video intake planning for the 3-class skeleton predictor."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key


@dataclass(frozen=True)
class CalibrationVideo:
    """One non-held-out MP4 candidate for v4 real-seed calibration."""

    source_path: Path
    label: str
    source_folder: str
    output_stem: str
    source_stem: str


def discover_calibration_videos(
    input_root: Path,
    *,
    heldout_roots: Sequence[Path] = (),
) -> list[CalibrationVideo]:
    """Discover final-label MP4s while rejecting held-out validation roots."""

    resolved_input = _resolved(input_root)
    for heldout_root in heldout_roots:
        if _is_within(resolved_input, _resolved(heldout_root)):
            raise ValueError(f"calibration input root overlaps held-out root: {input_root}")

    if not input_root.exists():
        raise FileNotFoundError(f"Calibration input root does not exist: {input_root}")

    videos: list[CalibrationVideo] = []
    for label in REVIEW_LABEL_ORDER:
        folder = input_root / label
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob("*.mp4"), key=natural_key)
        for index, path in enumerate(paths, start=1):
            resolved_path = _resolved(path)
            for heldout_root in heldout_roots:
                if _is_within(resolved_path, _resolved(heldout_root)):
                    raise ValueError(f"calibration video overlaps held-out root: {path}")
            videos.append(
                CalibrationVideo(
                    source_path=path,
                    label=label,
                    source_folder=label,
                    output_stem=f"v4_calibration_{label}_{index:06d}",
                    source_stem=path.stem,
                )
            )
    return videos


def validate_calibration_discovery(
    videos: Sequence[CalibrationVideo],
    *,
    expected_min_per_label: int = 10,
    allow_empty: bool = False,
) -> dict[str, object]:
    """Validate v4 calibration balance without allowing held-out leakage."""

    resolved = [_resolved(video.source_path) for video in videos]
    duplicate_count = len(resolved) - len(set(resolved))
    label_counts = Counter(video.label for video in videos)
    missing_or_low = {
        label: {
            "actual": int(label_counts.get(label, 0)),
            "minimum": int(expected_min_per_label),
        }
        for label in REVIEW_LABEL_ORDER
        if int(label_counts.get(label, 0)) < int(expected_min_per_label)
    }
    if not videos and allow_empty:
        status = "awaiting_calibration_videos"
    elif duplicate_count == 0 and not missing_or_low:
        status = "ready_for_skeleton_review"
    else:
        status = "insufficient_calibration_videos"
    return {
        "status": status,
        "passed": status == "ready_for_skeleton_review",
        "video_count": len(videos),
        "expected_min_per_label": int(expected_min_per_label),
        "duplicate_count": duplicate_count,
        "label_counts": dict(sorted(label_counts.items())),
        "missing_or_low_labels": missing_or_low,
    }


def build_v4_calibration_intake_report(
    *,
    input_root: Path,
    output_root: Path,
    heldout_roots: Sequence[Path],
    v3_summary_path: Path | None = None,
    expected_min_per_label: int = 10,
    allow_missing_input: bool = False,
) -> dict[str, object]:
    """Write v4 intake manifest and a recording plan."""

    output_root.mkdir(parents=True, exist_ok=True)
    if input_root.exists():
        videos = discover_calibration_videos(input_root, heldout_roots=heldout_roots)
    elif allow_missing_input:
        videos = []
    else:
        raise FileNotFoundError(f"Calibration input root does not exist: {input_root}")

    validation = validate_calibration_discovery(
        videos,
        expected_min_per_label=expected_min_per_label,
        allow_empty=allow_missing_input,
    )
    v3_summary = _load_json(v3_summary_path) if v3_summary_path is not None and v3_summary_path.exists() else {}
    failure_targets = build_failure_targets_from_v3(v3_summary, expected_min_per_label=expected_min_per_label)
    manifest = {
        "status": validation["status"],
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "heldout_roots": [path.as_posix() for path in heldout_roots],
        "heldout_policy": "exclude_all_paths_under_heldout_roots",
        "validation": validation,
        "videos": [
            {
                "source_path": video.source_path.as_posix(),
                "label": video.label,
                "source_folder": video.source_folder,
                "output_stem": video.output_stem,
                "source_stem": video.source_stem,
            }
            for video in videos
        ],
        "failure_targets": failure_targets,
    }
    manifest_path = output_root / "intake_manifest.json"
    targets_path = output_root / "failure_targets.json"
    plan_path = output_root / "recording_plan.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    targets_path.write_text(json.dumps(failure_targets, indent=2, ensure_ascii=False), encoding="utf-8")
    plan_path.write_text(_recording_plan_markdown(manifest), encoding="utf-8")
    return {
        "status": validation["status"],
        "passed": bool(validation["passed"]),
        "input_root": input_root.as_posix(),
        "output_root": output_root.as_posix(),
        "intake_manifest": manifest_path.as_posix(),
        "failure_targets": targets_path.as_posix(),
        "recording_plan": plan_path.as_posix(),
        "video_count": len(videos),
        "label_counts": validation["label_counts"],
    }


def build_v4_skeleton_review_plan(
    *,
    intake_manifest_path: Path,
    output_root: Path,
    review_output_root: Path,
) -> dict[str, object]:
    """Write the next MediaPipe skeleton-review plan for approved v4 calibration intake."""

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_json(intake_manifest_path)
    validation = _mapping(manifest.get("validation"))
    failure_targets = _mapping(manifest.get("failure_targets"))
    if bool(validation.get("passed")):
        label_counts = _mapping(validation.get("label_counts"))
        per_label = {label: int(_number(label_counts.get(label), 0)) for label in REVIEW_LABEL_ORDER}
    else:
        class_targets = _mapping(failure_targets.get("class_targets"))
        per_label = {
            label: int(_number(_mapping(class_targets.get(label)).get("minimum_new_clips"), 0))
            for label in REVIEW_LABEL_ORDER
        }
    expected_per_label = min((count for count in per_label.values() if count > 0), default=0)
    expected_count = sum(per_label.values())
    input_root = str(manifest.get("input_root", ""))
    training_policy = (
        "Approved v4 calibration clips may be used as non-held-out training seeds after visual skeleton approval. "
        "The held-out test root remains excluded."
    )
    command = [
        "python",
        "-m",
        "embodied_rps.tools.extract_real_hand_skeleton_review",
        "--input-root",
        input_root,
        "--output-root",
        review_output_root.as_posix(),
        "--expected-count",
        str(expected_count),
        "--expected-per-label",
        str(expected_per_label),
        "--output-prefix",
        "v4_calibration",
        "--review-stage",
        "v4_calibration_skeleton_review",
        "--training-policy",
        training_policy,
    ]
    status = "ready_for_skeleton_review" if bool(validation.get("passed")) else str(manifest.get("status", "not_ready"))
    plan = {
        "status": status,
        "intake_manifest": intake_manifest_path.as_posix(),
        "review_output_root": review_output_root.as_posix(),
        "input_root": input_root,
        "expected_count": expected_count,
        "expected_per_label": expected_per_label,
        "per_label_targets": per_label,
        "heldout_roots": list(_sequence(manifest.get("heldout_roots"))),
        "training_policy": training_policy,
        "review_command": command,
        "review_command_text": _quote_command(command),
        "acceptance_checks": [
            "Discovery must pass with the expected per-label counts.",
            "MediaPipe detection coverage should be inspected per clip before training use.",
            "Every skeleton-only and side-by-side MP4 must open and match the source frame count and resolution.",
            "Visual approval is required before v4 dataset generation.",
        ],
    }
    json_path = output_root / "skeleton_review_plan.json"
    md_path = output_root / "skeleton_review_plan.md"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_skeleton_review_plan_markdown(plan), encoding="utf-8")
    return {
        "status": status,
        "skeleton_review_plan": json_path.as_posix(),
        "skeleton_review_plan_md": md_path.as_posix(),
        "review_output_root": review_output_root.as_posix(),
        "expected_count": expected_count,
        "expected_per_label": expected_per_label,
    }


def build_v4_dataset_generation_plan(
    *,
    skeleton_review_plan_path: Path,
    output_root: Path,
    dataset_output_root: Path,
    base_dataset_root: Path,
    calibration_seed_package_root: Path | None = None,
    review_manifest_path: Path | None = None,
    min_detection_coverage: float = 0.98,
) -> dict[str, object]:
    """Validate v4 skeleton-review readiness before dataset generation."""

    output_root.mkdir(parents=True, exist_ok=True)
    skeleton_plan = _load_json(skeleton_review_plan_path)
    candidate_manifest_path = review_manifest_path or Path(str(skeleton_plan.get("review_output_root", ""))) / "manifest.json"
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    review_manifest: Mapping[str, object] = {}

    if not candidate_manifest_path.exists():
        status = "awaiting_skeleton_review"
        failures.append(
            {
                "code": "missing_review_manifest",
                "message": f"Review manifest does not exist: {candidate_manifest_path}",
            }
        )
    else:
        review_manifest = _load_json(candidate_manifest_path)
        failures.extend(
            validate_v4_review_manifest_for_dataset(
                review_manifest,
                skeleton_plan=skeleton_plan,
                min_detection_coverage=min_detection_coverage,
            )
        )
        status = "ready_for_v4_dataset_generation" if not failures else "review_not_ready_for_dataset"

    seed_package_root = calibration_seed_package_root or output_root.parent / "real_skeleton_v4_calibration_seed_package_20260612"
    generation_command = [
        "python",
        "-m",
        "embodied_rps.tools.generate_three_class_wait_skeleton_dataset",
        "--base-dataset-root",
        base_dataset_root.as_posix(),
        "--output-root",
        dataset_output_root.as_posix(),
        "--generated-per-target",
        "10000",
        "--augmentation-profile",
        "v3_targeted",
        "--calibration-seed-package-root",
        seed_package_root.as_posix(),
        "--overwrite",
    ]
    plan = {
        "status": status,
        "skeleton_review_plan": skeleton_review_plan_path.as_posix(),
        "review_manifest": candidate_manifest_path.as_posix(),
        "dataset_output_root": dataset_output_root.as_posix(),
        "base_dataset_root": base_dataset_root.as_posix(),
        "calibration_seed_package_root": seed_package_root.as_posix(),
        "min_detection_coverage": min_detection_coverage,
        "failures": failures,
        "warnings": warnings,
        "review_summary": _review_summary(review_manifest),
        "next_dataset_contract": {
            "implementation_required": "Build a passed v4 calibration seed package, then attach those real seed samples before procedural expansion.",
            "base_dataset_root": base_dataset_root.as_posix(),
            "approved_review_manifest": candidate_manifest_path.as_posix(),
            "calibration_seed_package_root": seed_package_root.as_posix(),
            "planned_dataset_output_root": dataset_output_root.as_posix(),
            "planned_profile_config": "configs/real_skeleton_three_class_wait_prediction_v4.yaml",
            "planned_generation_command": generation_command,
            "planned_generation_command_text": _quote_command(generation_command),
            "must_preserve_heldout_15": True,
        },
        "notes": [
            "This guard does not train a model.",
            "The held-out 15 MP4s remain excluded.",
            "If the status is ready_for_v4_dataset_generation, build the seed package from the approved skeleton review and then run the planned generation command.",
        ],
    }
    json_path = output_root / "dataset_generation_plan.json"
    md_path = output_root / "dataset_generation_plan.md"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_dataset_generation_plan_markdown(plan), encoding="utf-8")
    return {
        "status": status,
        "dataset_generation_plan": json_path.as_posix(),
        "dataset_generation_plan_md": md_path.as_posix(),
        "failure_count": len(failures),
        "dataset_output_root": dataset_output_root.as_posix(),
    }


def validate_v4_review_manifest_for_dataset(
    review_manifest: Mapping[str, object],
    *,
    skeleton_plan: Mapping[str, object],
    min_detection_coverage: float = 0.98,
) -> list[dict[str, object]]:
    """Return blocking issues before using v4 review outputs as training seeds."""

    failures: list[dict[str, object]] = []
    expected_count = int(_number(skeleton_plan.get("expected_count"), 0))
    expected_per_label = int(_number(skeleton_plan.get("expected_per_label"), 0))
    heldout_roots = [Path(str(path)) for path in _sequence(skeleton_plan.get("heldout_roots"))]
    if review_manifest.get("status") != "passed":
        failures.append({"code": "review_status_not_passed", "actual": review_manifest.get("status")})
    if review_manifest.get("review_stage") != "v4_calibration_skeleton_review":
        failures.append({"code": "unexpected_review_stage", "actual": review_manifest.get("review_stage")})
    training_policy = str(review_manifest.get("training_policy", ""))
    if "held-out" not in training_policy or "non-held-out" not in training_policy:
        failures.append({"code": "training_policy_does_not_preserve_heldout_exclusion", "actual": training_policy})
    discovery = _mapping(review_manifest.get("discovery"))
    if not bool(discovery.get("passed")):
        failures.append({"code": "discovery_not_passed", "discovery": dict(discovery)})
    label_counts = _mapping(discovery.get("label_counts"))
    for label in REVIEW_LABEL_ORDER:
        actual = int(_number(label_counts.get(label), 0))
        if expected_per_label and actual != expected_per_label:
            failures.append({"code": "label_count_mismatch", "label": label, "actual": actual, "expected": expected_per_label})
    video_count = int(_number(review_manifest.get("video_count"), 0))
    if expected_count and video_count != expected_count:
        failures.append({"code": "video_count_mismatch", "actual": video_count, "expected": expected_count})
    validation = _mapping(review_manifest.get("validation"))
    if validation.get("status") != "passed":
        failures.append({"code": "output_video_validation_not_passed", "validation_status": validation.get("status")})
    records = _mapping_sequence(review_manifest.get("records"))
    if expected_count and len(records) != expected_count:
        failures.append({"code": "record_count_mismatch", "actual": len(records), "expected": expected_count})
    for record in records:
        video_id = str(record.get("video_id", ""))
        source_path = Path(str(record.get("source_path", "")))
        for heldout_root in heldout_roots:
            if _is_within(_resolved(source_path), _resolved(heldout_root)):
                failures.append({"code": "heldout_source_leak", "video_id": video_id, "source_path": source_path.as_posix()})
        coverage = _number(record.get("detection_coverage"), 0.0)
        missing_frames = int(_number(record.get("missing_frames"), 0))
        needs_review = bool(record.get("needs_review"))
        if coverage < min_detection_coverage:
            failures.append({"code": "low_detection_coverage", "video_id": video_id, "actual": coverage, "minimum": min_detection_coverage})
        if missing_frames > 0:
            failures.append({"code": "missing_detection_frames", "video_id": video_id, "missing_frames": missing_frames})
        if needs_review:
            failures.append({"code": "record_flagged_needs_review", "video_id": video_id})
    return failures


def build_failure_targets_from_v3(
    v3_summary: Mapping[str, object],
    *,
    expected_min_per_label: int = 10,
) -> dict[str, object]:
    """Derive v4 recording priorities from v3 validation failures."""

    gates = _mapping(v3_summary.get("mp4_gates"))
    tcn_new15 = _mapping(gates.get("tcn_new15"))
    per_class = _mapping(tcn_new15.get("per_class"))
    failed_focus = _mapping(v3_summary.get("failed_clip_focus"))
    tcn_original_failures = list(_mapping_sequence(failed_focus.get("tcn_original20")))
    tcn_new_failures = list(_mapping_sequence(failed_focus.get("tcn_new15")))

    late_original_paper = [
        item
        for item in tcn_original_failures
        if _string(item.get("true_gesture")) == "paper" and _string(item.get("failure_reason")) == "late_decision"
    ]
    heldout_failure_counts = Counter(_string(item.get("failure_reason")) for item in tcn_new_failures)
    heldout_pred_counts = Counter(_string(item.get("predicted_gesture")) for item in tcn_new_failures)
    rock_false_triggers = int(_number(tcn_new15.get("rock_false_trigger_count"), 0))
    rock_wait_success = int(_number(tcn_new15.get("rock_wait_success_count"), 0))
    rock_count = int(_number(tcn_new15.get("rock_clip_count"), 0))

    return {
        "source": "v3_training_and_validation_summary",
        "do_not_use_heldout_15_as_training": True,
        "minimum_per_label": int(expected_min_per_label),
        "class_targets": {
            "rock": {
                "minimum_new_clips": max(int(expected_min_per_label), 20 if rock_false_triggers else int(expected_min_per_label)),
                "priority": "high" if rock_false_triggers or rock_wait_success < rock_count else "medium",
                "reasons": [
                    f"held-out rock wait success was {rock_wait_success}/{rock_count}",
                    f"held-out rock false triggers were {rock_false_triggers}",
                    "record no-transition rock holds with small wrist/view changes and varied backgrounds",
                ],
            },
            "paper": {
                "minimum_new_clips": max(int(expected_min_per_label), 20 if late_original_paper else int(expected_min_per_label)),
                "priority": "high" if late_original_paper else "medium",
                "reasons": [
                    f"late original paper failures: {len(late_original_paper)}",
                    "record slow and hesitant rock-to-paper openings that become distinguishable before progress 0.50",
                ],
            },
            "scissors": {
                "minimum_new_clips": max(int(expected_min_per_label), 20),
                "priority": "high",
                "reasons": [
                    f"held-out scissors pass count was {_class_pass_count(per_class, 'scissors')}/{_class_clip_count(per_class, 'scissors')}",
                    "record viewpoint-shaky scissors with palm roll wobble, camera jitter, and different hand scales",
                ],
            },
        },
        "heldout_failure_reason_counts": dict(sorted(heldout_failure_counts.items())),
        "heldout_failed_prediction_counts": dict(sorted(heldout_pred_counts.items())),
    }


def _recording_plan_markdown(manifest: Mapping[str, object]) -> str:
    targets = _mapping(manifest.get("failure_targets"))
    class_targets = _mapping(targets.get("class_targets"))
    lines = [
        "# V4 Calibration Recording Plan",
        "",
        "## Status",
        "",
        f"- Intake status: `{manifest.get('status')}`",
        f"- Input root: `{manifest.get('input_root')}`",
        "- Held-out policy: do not copy or train on any MP4 under the held-out roots.",
        "",
        "## Required Folder Layout",
        "",
        "```text",
        "<calibration_root>/",
        "  rock/*.mp4",
        "  paper/*.mp4",
        "  scissors/*.mp4",
        "```",
        "",
        "## Recording Targets",
        "",
    ]
    for label in REVIEW_LABEL_ORDER:
        target = _mapping(class_targets.get(label))
        reasons = [str(item) for item in _sequence(target.get("reasons"))]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- Minimum new non-held-out clips: `{target.get('minimum_new_clips')}`")
        lines.append(f"- Priority: `{target.get('priority')}`")
        for reason in reasons:
            lines.append(f"- {reason}")
        lines.append("")
    lines.extend(
        [
            "## Next Command",
            "",
            "After recording the non-held-out calibration MP4s, run the MediaPipe skeleton review on this calibration root, then build v4 shards from approved skeletons plus procedural perturbations.",
            "",
        ]
    )
    return "\n".join(lines)


def _skeleton_review_plan_markdown(plan: Mapping[str, object]) -> str:
    lines = [
        "# V4 Calibration Skeleton Review Plan",
        "",
        "## Status",
        "",
        f"- Plan status: `{plan.get('status')}`",
        f"- Input root: `{plan.get('input_root')}`",
        f"- Review output root: `{plan.get('review_output_root')}`",
        f"- Expected clips: `{plan.get('expected_count')}`",
        f"- Expected clips per label: `{plan.get('expected_per_label')}`",
        "",
        "## Command",
        "",
        "```powershell",
        str(plan.get("review_command_text", "")),
        "```",
        "",
        "## Acceptance Checks",
        "",
    ]
    for check in _sequence(plan.get("acceptance_checks")):
        lines.append(f"- {check}")
    lines.extend(
        [
            "",
            "## Training Policy",
            "",
            str(plan.get("training_policy", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _dataset_generation_plan_markdown(plan: Mapping[str, object]) -> str:
    lines = [
        "# V4 Dataset Generation Readiness Plan",
        "",
        "## Status",
        "",
        f"- Status: `{plan.get('status')}`",
        f"- Review manifest: `{plan.get('review_manifest')}`",
        f"- Dataset output root: `{plan.get('dataset_output_root')}`",
        f"- Minimum detection coverage: `{plan.get('min_detection_coverage')}`",
        "",
        "## Blocking Issues",
        "",
    ]
    failures = list(_sequence(plan.get("failures")))
    if failures:
        for failure in failures:
            lines.append(f"- `{_mapping(failure).get('code')}`: `{json.dumps(failure, ensure_ascii=False)}`")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Next Dataset Contract",
            "",
            f"- Calibration seed package root: `{_mapping(plan.get('next_dataset_contract')).get('calibration_seed_package_root')}`",
            f"- Planned profile config: `{_mapping(plan.get('next_dataset_contract')).get('planned_profile_config')}`",
            "- Planned generation command after the seed package passes:",
            "",
            "```powershell",
            str(_mapping(plan.get("next_dataset_contract")).get("planned_generation_command_text", "")),
            "```",
            "",
            "## Notes",
            "",
        ]
    )
    for note in _sequence(plan.get("notes")):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _review_summary(review_manifest: Mapping[str, object]) -> dict[str, object]:
    if not review_manifest:
        return {}
    discovery = _mapping(review_manifest.get("discovery"))
    records = _mapping_sequence(review_manifest.get("records"))
    coverages = [_number(record.get("detection_coverage"), 0.0) for record in records]
    return {
        "status": review_manifest.get("status"),
        "review_stage": review_manifest.get("review_stage"),
        "video_count": review_manifest.get("video_count"),
        "label_counts": dict(_mapping(discovery.get("label_counts"))),
        "detection_coverage_min": min(coverages) if coverages else None,
        "detection_coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        "needs_review_count": sum(1 for record in records if bool(record.get("needs_review"))),
    }


def _load_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)


def _mapping_sequence(value: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string(value: object) -> str:
    return str(value) if value is not None else ""


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_pass_count(per_class: Mapping[str, object], label: str) -> int:
    return int(_number(_mapping(per_class.get(label)).get("passed_count"), 0))


def _class_clip_count(per_class: Mapping[str, object], label: str) -> int:
    return int(_number(_mapping(per_class.get(label)).get("clip_count"), 0))


def _quote_command(parts: Sequence[str]) -> str:
    quoted: list[str] = []
    for part in parts:
        if not part or any(char.isspace() for char in part):
            quoted.append(f"'{part}'")
        else:
            quoted.append(part)
    return " ".join(quoted)


__all__ = [
    "CalibrationVideo",
    "build_failure_targets_from_v3",
    "build_v4_calibration_intake_report",
    "build_v4_dataset_generation_plan",
    "build_v4_skeleton_review_plan",
    "discover_calibration_videos",
    "validate_v4_review_manifest_for_dataset",
    "validate_calibration_discovery",
]
