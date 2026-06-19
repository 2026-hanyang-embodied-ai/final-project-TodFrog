"""Evidence summary helpers for SCHUNK Isaac render outputs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


GESTURES = ("rock", "paper", "scissors")


def summarize_schunk_visual_evidence(
    *,
    schunk_render_dir: Path,
    visual_rig_render_dir: Path,
    out_json: Path | None = None,
    out_markdown: Path | None = None,
) -> dict[str, Any]:
    """Summarize mesh and visual-rig render evidence without mixing claims."""

    schunk_mesh = _summarize_render_dir(
        schunk_render_dir,
        expected_visual_evidence="schunk_mesh",
        label="imported_schunk_mesh",
    )
    visual_rig = _summarize_render_dir(
        visual_rig_render_dir,
        expected_visual_evidence="procedural_visual_rig",
        label="procedural_visual_rig",
    )
    schunk_mesh_passed = _evidence_passed(schunk_mesh, "schunk_mesh")
    visual_rig_passed = _evidence_passed(visual_rig, "procedural_visual_rig")
    if schunk_mesh_passed:
        overall_status = "passed_schunk_mesh"
    elif visual_rig_passed:
        overall_status = "passed_labeled_visual_rig_only"
    else:
        overall_status = "failed"

    safe_claims: list[str] = []
    blocked_claims: list[str] = []
    if schunk_mesh_passed:
        safe_claims.append("Imported SCHUNK mesh render evidence passed the schunk_mesh postcondition.")
    else:
        blocked_claims.append("Do not claim that the imported dex-urdf SCHUNK mesh visibly forms rock, paper, and scissors.")
    if visual_rig_passed:
        safe_claims.append(
            "The Isaac-side procedural visual rig provides labeled, readable RPS and fist-start sequence visualization."
        )
    if _count_records_with_status(schunk_mesh, "debug_skeleton_overlay", "applied") > 0:
        blocked_claims.append("2D debug-overlay outputs are diagnostic only and cannot satisfy SCHUNK mesh evidence.")

    summary: dict[str, Any] = {
        "overall_status": overall_status,
        "schunk_mesh": schunk_mesh,
        "procedural_visual_rig": visual_rig,
        "safe_claims": safe_claims,
        "blocked_claims": blocked_claims,
        "next_actions": [
            "Use procedural visual-rig images only with an explicit visualization label.",
            "Keep imported SCHUNK mesh-only RPS visual proof marked unresolved until a clean mesh articulation path passes.",
            "For physical-fidelity claims, evaluate a cleaner Isaac hand asset or build a controlled segmented SCHUNK-like mesh rig.",
        ],
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if out_markdown is not None:
        out_markdown.parent.mkdir(parents=True, exist_ok=True)
        out_markdown.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _summarize_render_dir(render_dir: Path, *, expected_visual_evidence: str, label: str) -> dict[str, Any]:
    postcondition = _read_json_mapping(render_dir / "render_postcondition.json")
    diagnostics = _read_json_mapping(render_dir / "render_diagnostics.json")
    records = _read_json_list(render_dir / "render_records.json")
    records_summary = _summarize_records(records)
    return {
        "label": label,
        "render_dir": render_dir.as_posix(),
        "expected_visual_evidence": expected_visual_evidence,
        "postcondition_status": _string_or_none(postcondition.get("status")),
        "postcondition_error": _string_or_none(postcondition.get("error")),
        "postcondition_required_visual_evidence": _string_or_none(postcondition.get("required_visual_evidence")),
        "articulation_status": _string_or_none(diagnostics.get("articulation_status")),
        "render_validation_status": _string_or_none(diagnostics.get("render_validation_status")),
        "dof_count": _int_or_none(diagnostics.get("dof_count")) or _list_length(diagnostics.get("dof_names")),
        "sequence_render": diagnostics.get("sequence_render") if isinstance(diagnostics.get("sequence_render"), Mapping) else None,
        "records": records_summary,
    }


def _summarize_records(records: list[Any]) -> dict[str, Any]:
    valid_records = [record for record in records if isinstance(record, Mapping)]
    record_type_counts: Counter[str] = Counter()
    visual_sync_counts: Counter[str] = Counter()
    procedural_counts: Counter[str] = Counter()
    overlay_counts: Counter[str] = Counter()
    gesture_counts: Counter[str] = Counter()
    sequence_frame_counts: Counter[str] = Counter()
    max_joint_error: float | None = None
    static_records: list[Mapping[str, Any]] = []
    for record in valid_records:
        gesture = str(record.get("gesture", "unknown"))
        gesture_counts[gesture] += 1
        record_type = str(record.get("record_type", "static"))
        record_type_counts[record_type] += 1
        visual_sync_counts[_status_from(record, "visual_sync")] += 1
        procedural_counts[_status_from(record, "procedural_visual_rig")] += 1
        overlay_counts[_status_from(record, "debug_skeleton_overlay")] += 1
        if record_type == "sequence_frame":
            sequence_frame_counts[gesture] += 1
        else:
            static_records.append(record)
        joint_error = _float_or_none(record.get("max_abs_joint_error_degrees"))
        if joint_error is not None:
            max_joint_error = joint_error if max_joint_error is None else max(max_joint_error, joint_error)
    return {
        "record_count": len(valid_records),
        "record_type_counts": dict(sorted(record_type_counts.items())),
        "gesture_counts": dict(sorted(gesture_counts.items())),
        "visual_sync_status_counts": dict(sorted(visual_sync_counts.items())),
        "procedural_visual_rig_status_counts": dict(sorted(procedural_counts.items())),
        "debug_skeleton_overlay_status_counts": dict(sorted(overlay_counts.items())),
        "sequence_frame_counts_by_gesture": dict(sorted(sequence_frame_counts.items())),
        "representative_static_images": _representative_images(static_records),
        "max_abs_joint_error_degrees": max_joint_error,
    }


def _representative_images(records: list[Mapping[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for gesture in GESTURES:
        candidates = [record for record in records if record.get("gesture") == gesture and isinstance(record.get("image_path"), str)]
        if not candidates:
            continue
        preferred = [
            record
            for record in candidates
            if _float_or_none(record.get("yaw_degrees")) == 45.0 and _float_or_none(record.get("pitch_degrees")) == 20.0
        ]
        chosen = preferred[0] if preferred else candidates[0]
        selected[gesture] = str(chosen["image_path"])
    return selected


def _evidence_passed(summary: Mapping[str, Any], expected_visual_evidence: str) -> bool:
    return (
        summary.get("postcondition_status") == "passed"
        and summary.get("postcondition_required_visual_evidence") == expected_visual_evidence
        and summary.get("articulation_status") == "initialized"
        and summary.get("render_validation_status") == "passed"
    )


def _count_records_with_status(summary: Mapping[str, Any], field_name: str, status: str) -> int:
    records = summary.get("records")
    if not isinstance(records, Mapping):
        return 0
    counts = records.get(f"{field_name}_status_counts")
    if not isinstance(counts, Mapping):
        return 0
    value = counts.get(status, 0)
    return int(value) if isinstance(value, int) else 0


def _status_from(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if isinstance(value, Mapping):
        status = value.get("status")
        return str(status) if isinstance(status, str) and status != "" else "missing"
    return "missing"


def _render_markdown(summary: Mapping[str, Any]) -> str:
    schunk_mesh = _mapping(summary.get("schunk_mesh"))
    visual_rig = _mapping(summary.get("procedural_visual_rig"))
    lines = [
        "# SCHUNK Visual Evidence Summary",
        "",
        f"- Overall status: `{summary.get('overall_status')}`",
        f"- Imported SCHUNK mesh postcondition: `{schunk_mesh.get('postcondition_status')}`",
        f"- Procedural visual rig postcondition: `{visual_rig.get('postcondition_status')}`",
        "",
        "## Evidence Table",
        "",
        "| Evidence path | Required evidence | Status | Records | Images |",
        "| --- | --- | --- | ---: | ---: |",
        _markdown_row(schunk_mesh),
        _markdown_row(visual_rig),
        "",
        "## Safe Claims",
        "",
    ]
    safe_claims = summary.get("safe_claims")
    if isinstance(safe_claims, list) and safe_claims:
        lines.extend(f"- {claim}" for claim in safe_claims)
    else:
        lines.append("- No visual evidence claim is currently safe.")
    lines.extend(["", "## Blocked Claims", ""])
    blocked_claims = summary.get("blocked_claims")
    if isinstance(blocked_claims, list) and blocked_claims:
        lines.extend(f"- {claim}" for claim in blocked_claims)
    else:
        lines.append("- No blocked visual claim recorded.")
    lines.extend(["", "## Representative Images", ""])
    for label, evidence in (("Imported SCHUNK mesh", schunk_mesh), ("Procedural visual rig", visual_rig)):
        lines.append(f"### {label}")
        records = _mapping(evidence.get("records"))
        images = records.get("representative_static_images")
        if isinstance(images, Mapping) and images:
            for gesture in GESTURES:
                image = images.get(gesture)
                if isinstance(image, str):
                    lines.append(f"- `{gesture}`: `{image}`")
        else:
            lines.append("- No representative static image recorded.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_row(evidence: Mapping[str, Any]) -> str:
    records = _mapping(evidence.get("records"))
    record_count = records.get("record_count")
    image_count = record_count if isinstance(record_count, int) else 0
    return (
        f"| `{evidence.get('label')}` | `{evidence.get('expected_visual_evidence')}` | "
        f"`{evidence.get('postcondition_status')}` | {int(image_count)} | {int(image_count)} |"
    )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": path.as_posix()}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        return {"status": "invalid", "path": path.as_posix()}
    return {str(key): value for key, value in loaded.items()}


def _read_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, list) else []


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _list_length(value: object) -> int | None:
    return len(value) if isinstance(value, list) else None
