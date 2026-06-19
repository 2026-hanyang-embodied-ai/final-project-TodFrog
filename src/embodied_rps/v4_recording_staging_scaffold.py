"""Folder scaffold for raw v4 recording staging clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER


@dataclass(frozen=True)
class V4RecordingStagingScaffoldConfig:
    """Configuration for creating the v4 recording staging layout."""

    staging_root: Path
    calibration_root: Path
    heldout_roots: tuple[Path, ...] = ()
    expected_per_label: int = 20


def prepare_v4_recording_staging_scaffold(config: V4RecordingStagingScaffoldConfig) -> dict[str, object]:
    """Create label folders and guidance for raw v4 staging MP4s."""

    _validate_config(config)
    config.staging_root.mkdir(parents=True, exist_ok=True)
    for label in REVIEW_LABEL_ORDER:
        (config.staging_root / label).mkdir(exist_ok=True)
    checklist = _checklist(config)
    readme = _readme(config, checklist)
    checklist_path = config.staging_root / "staging_checklist.json"
    readme_path = config.staging_root / "README.md"
    checklist_path.write_text(json.dumps(checklist, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")
    return {
        "status": "ready_for_staging",
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "expected_per_label": config.expected_per_label,
        "expected_total": config.expected_per_label * len(REVIEW_LABEL_ORDER),
        "label_dirs": {label: (config.staging_root / label).as_posix() for label in REVIEW_LABEL_ORDER},
        "readme": readme_path.as_posix(),
        "staging_checklist": checklist_path.as_posix(),
        "next_command": (
            "python -m embodied_rps.tools.run_v4_recording_ingest "
            "--source-root <staging_root> "
            "--calibration-root <calibration_root> "
            "--heldout-root <heldout_test_root> "
            "--output-root artifacts/real_skeleton_v4_recording_ingest_20260612"
        ),
    }


def _validate_config(config: V4RecordingStagingScaffoldConfig) -> None:
    if config.expected_per_label <= 0:
        raise ValueError("expected_per_label must be positive")
    resolved_staging = _resolved(config.staging_root)
    resolved_calibration = _resolved(config.calibration_root)
    if _overlaps(resolved_staging, resolved_calibration):
        raise ValueError(f"staging root overlaps calibration root: {config.staging_root} vs {config.calibration_root}")
    for heldout_root in config.heldout_roots:
        resolved_heldout = _resolved(heldout_root)
        if _overlaps(resolved_staging, resolved_heldout):
            raise ValueError(f"staging root overlaps held-out root: {config.staging_root} vs {heldout_root}")


def _checklist(config: V4RecordingStagingScaffoldConfig) -> dict[str, object]:
    labels = {
        "rock": {
            "target_new_clips": config.expected_per_label,
            "recording_focus": [
                "No-transition rock holds with small wrist/view changes.",
                "Include different backgrounds, hand sizes, and distances.",
                "Keep the final opponent gesture as rock.",
            ],
        },
        "paper": {
            "target_new_clips": config.expected_per_label,
            "recording_focus": [
                "Slow or hesitant rock-to-paper transitions.",
                "Long fist-like early hold before finger opening.",
                "Stagger ring and pinky timing where natural.",
            ],
        },
        "scissors": {
            "target_new_clips": config.expected_per_label,
            "recording_focus": [
                "Clear final scissors under varied yaw, pitch, and roll.",
                "Mild palm-roll wobble and camera-like jitter.",
                "Do not let shaky clips obscure the final class.",
            ],
        },
    }
    return {
        "status": "ready_for_staging",
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix(),
        "heldout_roots": [path.as_posix() for path in config.heldout_roots],
        "heldout_policy": "do_not_copy_or_train_on_heldout_test_clips",
        "copy_policy": "dry_run_assignment_first_then_execute_copy",
        "expected_per_label": config.expected_per_label,
        "expected_total": config.expected_per_label * len(REVIEW_LABEL_ORDER),
        "labels": labels,
    }


def _readme(config: V4RecordingStagingScaffoldConfig, checklist: dict[str, object]) -> str:
    lines = [
        "# V4 Recording Staging Folder",
        "",
        "This folder is for raw, newly recorded, non-held-out MP4s before they are copied into planned calibration slots.",
        "",
        "Do not copy any clip from the held-out test set into this folder.",
        "",
        "## Required Layout",
        "",
        "```text",
        "v4_recording_staging/",
        "  rock/*.mp4",
        "  paper/*.mp4",
        "  scissors/*.mp4",
        "```",
        "",
        "## Target Counts",
        "",
        f"- Target per label: `{config.expected_per_label}`",
        f"- Target total: `{checklist['expected_total']}`",
        "",
        "## Recording Focus",
        "",
        "- Rock: no-transition holds with small wrist/view changes.",
        "- Paper: slow or hesitant openings after a long fist-like start.",
        "- Scissors: varied viewpoints, mild wobble, and clear final scissors.",
        "",
        "## Next Step",
        "",
        "Run the v4 recording ingest command in dry-run mode, review the assignment table, then rerun with `--execute-copy` only after the mapping is correct.",
        "",
    ]
    return "\n".join(lines)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _overlaps(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = ["V4RecordingStagingScaffoldConfig", "prepare_v4_recording_staging_scaffold"]
