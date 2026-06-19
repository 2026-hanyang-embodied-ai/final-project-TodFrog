"""Deterministic recording slot manifest for v4 calibration videos."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER


@dataclass(frozen=True)
class V4RecordingSlotManifestConfig:
    """Configuration for the v4 recording slot manifest."""

    calibration_root: Path
    expected_per_label: int = 20


def write_v4_recording_slot_manifest(config: V4RecordingSlotManifestConfig) -> dict[str, object]:
    """Write deterministic recording slot targets for the v4 calibration set."""

    if config.expected_per_label <= 0:
        raise ValueError("expected_per_label must be positive")
    config.calibration_root.mkdir(parents=True, exist_ok=True)
    slots = _build_slots(config)
    json_path = config.calibration_root / "recording_slot_manifest.json"
    csv_path = config.calibration_root / "recording_slot_manifest.csv"
    md_path = config.calibration_root / "recording_slot_manifest.md"
    json_path.write_text(json.dumps(slots, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path.write_text(_slots_csv(slots), encoding="utf-8", newline="")
    md_path.write_text(_slots_markdown(config, slots), encoding="utf-8")
    return {
        "status": "ready_for_recording_slots",
        "calibration_root": config.calibration_root.as_posix(),
        "expected_per_label": int(config.expected_per_label),
        "expected_total": int(config.expected_per_label) * len(REVIEW_LABEL_ORDER),
        "slot_count": len(slots),
        "label_counts": {label: sum(1 for slot in slots if slot["label"] == label) for label in REVIEW_LABEL_ORDER},
        "json_path": json_path.as_posix(),
        "csv_path": csv_path.as_posix(),
        "markdown_path": md_path.as_posix(),
    }


def _build_slots(config: V4RecordingSlotManifestConfig) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    viewpoints = ("front", "left_oblique", "right_oblique", "top_down", "low_angle")
    backgrounds = ("plain_light", "plain_dark", "desk_clutter", "monitor_background", "room_background")
    distances = ("close", "medium", "far")
    handedness = ("right", "left_or_mirrored", "either")
    speeds = ("slow", "normal", "fast", "hesitant")
    stability = ("steady", "mild_camera_jitter", "mild_palm_roll", "small_depth_change")
    for label in REVIEW_LABEL_ORDER:
        for index in range(config.expected_per_label):
            one_based = index + 1
            filename = f"{label}_{one_based:03d}.mp4"
            slots.append(
                {
                    "slot_id": f"v4_{label}_{one_based:03d}",
                    "label": label,
                    "filename": filename,
                    "target_path": (config.calibration_root / label / filename).as_posix(),
                    "viewpoint": viewpoints[index % len(viewpoints)],
                    "background": backgrounds[(index // len(viewpoints)) % len(backgrounds)],
                    "distance": distances[index % len(distances)],
                    "handedness_target": handedness[(index // 2) % len(handedness)],
                    "speed_focus": speeds[(index + _label_offset(label)) % len(speeds)],
                    "stability_focus": stability[(index + 2 * _label_offset(label)) % len(stability)],
                    "motion_focus": _motion_focus(label, index),
                    "review_focus": _review_focus(label),
                    "heldout_policy": "new_non_heldout_recording_only",
                }
            )
    return slots


def _motion_focus(label: str, index: int) -> str:
    families = {
        "rock": (
            "rock_hold_no_transition",
            "rock_hold_small_wrist_change",
            "rock_hold_small_depth_change",
            "rock_hold_low_transition_mass",
        ),
        "paper": (
            "hard_paper_long_fist_hold",
            "hard_paper_slow_opening",
            "hard_paper_staggered_ring_pinky",
            "hard_paper_hesitant_opening",
        ),
        "scissors": (
            "shaky_scissors_clear_final",
            "scissors_palm_roll_wobble",
            "scissors_viewpoint_shift",
            "scissors_fingertip_mcp_drift",
        ),
    }
    return families[label][index % len(families[label])]


def _review_focus(label: str) -> str:
    if label == "rock":
        return "Should select wait_counter_paper by progress <= 0.50 without paper/scissors false trigger."
    if label == "paper":
        return "Early segment may remain fist-like, but final paper must be clear."
    return "Final scissors must stay clear despite mild shake or viewpoint change."


def _label_offset(label: str) -> int:
    return {"rock": 0, "paper": 1, "scissors": 2}[label]


def _slots_csv(slots: list[dict[str, object]]) -> str:
    output = StringIO()
    fieldnames = [
        "slot_id",
        "label",
        "filename",
        "target_path",
        "viewpoint",
        "background",
        "distance",
        "handedness_target",
        "speed_focus",
        "stability_focus",
        "motion_focus",
        "review_focus",
        "heldout_policy",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(slots)
    return output.getvalue()


def _slots_markdown(config: V4RecordingSlotManifestConfig, slots: list[dict[str, object]]) -> str:
    lines = [
        "# V4 Recording Slot Manifest",
        "",
        "These slots define the first non-held-out v4 calibration recording target.",
        "",
        f"- Calibration root: `{config.calibration_root.as_posix()}`",
        f"- Required per label: `{config.expected_per_label}`",
        f"- Required total: `{len(slots)}`",
        "- Held-out `test` clips must not be copied into these slots.",
        "",
        "## Slot Table",
        "",
        "| Slot | Label | File | View | Background | Speed | Stability | Motion focus |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for slot in slots:
        lines.append(
            "| `{slot_id}` | `{label}` | `{filename}` | `{viewpoint}` | `{background}` | `{speed_focus}` | `{stability_focus}` | `{motion_focus}` |".format(
                **slot
            )
        )
    lines.extend(
        [
            "",
            "## Review Rule",
            "",
            "After recording these files, run MP4 preflight and MediaPipe skeleton review before any dataset generation.",
            "Use the review videos to reject clips with missing hands, unclear final gesture, or excessive camera shake.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["V4RecordingSlotManifestConfig", "write_v4_recording_slot_manifest"]
