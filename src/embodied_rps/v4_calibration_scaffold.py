"""Folder scaffold for non-held-out v4 calibration recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER
from embodied_rps.v4_recording_slot_manifest import V4RecordingSlotManifestConfig, write_v4_recording_slot_manifest


@dataclass(frozen=True)
class V4CalibrationScaffoldConfig:
    """Configuration for creating the v4 calibration recording folder layout."""

    calibration_root: Path
    heldout_roots: tuple[Path, ...] = ()
    expected_per_label: int = 20


def prepare_v4_calibration_scaffold(config: V4CalibrationScaffoldConfig) -> dict[str, object]:
    """Create class folders and recording guidance for the v4 calibration set."""

    _validate_config(config)
    config.calibration_root.mkdir(parents=True, exist_ok=True)
    for label in REVIEW_LABEL_ORDER:
        (config.calibration_root / label).mkdir(exist_ok=True)
    checklist = _checklist(config)
    readme = _readme(config, checklist)
    checklist_path = config.calibration_root / "recording_checklist.json"
    readme_path = config.calibration_root / "README.md"
    checklist_path.write_text(json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")
    slot_manifest = write_v4_recording_slot_manifest(
        V4RecordingSlotManifestConfig(
            calibration_root=config.calibration_root,
            expected_per_label=config.expected_per_label,
        )
    )
    return {
        "status": "ready_for_recording",
        "calibration_root": config.calibration_root.as_posix(),
        "expected_per_label": config.expected_per_label,
        "expected_total": config.expected_per_label * len(REVIEW_LABEL_ORDER),
        "label_dirs": {label: (config.calibration_root / label).as_posix() for label in REVIEW_LABEL_ORDER},
        "readme": readme_path.as_posix(),
        "recording_checklist": checklist_path.as_posix(),
        "recording_slot_manifest": slot_manifest,
    }


def _validate_config(config: V4CalibrationScaffoldConfig) -> None:
    if config.expected_per_label <= 0:
        raise ValueError("expected_per_label must be positive")
    resolved_root = _resolved(config.calibration_root)
    for heldout_root in config.heldout_roots:
        resolved_heldout = _resolved(heldout_root)
        if _is_within(resolved_root, resolved_heldout) or _is_within(resolved_heldout, resolved_root):
            raise ValueError(f"calibration root overlaps held-out root: {config.calibration_root} vs {heldout_root}")


def _checklist(config: V4CalibrationScaffoldConfig) -> dict[str, object]:
    labels = {
        "rock": {
            "minimum_new_clips": config.expected_per_label,
            "recording_focus": [
                "No-transition rock holds with small wrist/view changes.",
                "Different backgrounds and hand sizes.",
                "Avoid copying held-out test clips.",
            ],
        },
        "paper": {
            "minimum_new_clips": config.expected_per_label,
            "recording_focus": [
                "Slow or hesitant rock-to-paper openings.",
                "Fist-like early hold before opening.",
                "Ring and pinky delayed but visible before progress 0.50 when possible.",
            ],
        },
        "scissors": {
            "minimum_new_clips": config.expected_per_label,
            "recording_focus": [
                "Diverse camera viewpoints.",
                "Mild palm roll wobble and camera-like jitter.",
                "Final scissors must remain visually clear.",
            ],
        },
    }
    return {
        "status": "ready_for_recording",
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_roots": [path.as_posix() for path in config.heldout_roots],
        "heldout_policy": "do_not_copy_or_train_on_heldout_test_clips",
        "expected_per_label": config.expected_per_label,
        "expected_total": config.expected_per_label * len(REVIEW_LABEL_ORDER),
        "labels": labels,
        "next_command_after_recording": (
            "python -m embodied_rps.tools.prepare_v4_calibration_intake "
            "--input-root <calibration_root> "
            "--output-root artifacts/real_skeleton_v4_calibration_intake_20260611 "
            "--heldout-root <heldout_test_root> "
            "--expected-min-per-label 20"
        ),
    }


def _readme(config: V4CalibrationScaffoldConfig, checklist: dict[str, object]) -> str:
    lines = [
        "# V4 Calibration Recording Folder",
        "",
        "This folder is for new non-held-out calibration videos only.",
        "",
        "Do not copy any clip from the held-out test set into this folder.",
        "",
        "## Required Layout",
        "",
        "```text",
        "v4_calibration/",
        "  rock/*.mp4",
        "  paper/*.mp4",
        "  scissors/*.mp4",
        "```",
        "",
        "## Required Counts",
        "",
        f"- Minimum per label: `{config.expected_per_label}`",
        f"- Minimum total: `{checklist['expected_total']}`",
        "",
        "## Recording Focus",
        "",
        "- Rock: no-transition holds with small wrist/view changes.",
        "- Paper: slow or hesitant openings after an early fist-like hold.",
        "- Scissors: Diverse camera viewpoints, mild palm roll wobble, and clear final scissors.",
        "",
        "## Next Step",
        "",
        "After recording, run v4 intake and then MediaPipe skeleton review before using these clips as training seeds.",
        "",
    ]
    return "\n".join(lines)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = ["V4CalibrationScaffoldConfig", "prepare_v4_calibration_scaffold"]
