"""Audit raw v4 recording staging MP4s before slot assignment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import REVIEW_LABEL_ORDER, natural_key

VideoProbe = Callable[[Path], dict[str, object]]


@dataclass(frozen=True)
class V4RecordingStagingAuditConfig:
    """Configuration for auditing raw staging MP4s."""

    staging_root: Path
    output_root: Path
    expected_per_label: int = 20
    calibration_root: Path | None = None
    heldout_roots: tuple[Path, ...] = ()
    min_frame_count: int = 5
    min_fps: float = 1.0
    min_width: int = 1
    min_height: int = 1


def audit_v4_recording_staging(
    config: V4RecordingStagingAuditConfig,
    *,
    video_probe: VideoProbe | None = None,
) -> dict[str, object]:
    """Audit staging folder counts, labels, file size, and leakage risks."""

    if video_probe is None:
        from embodied_rps.v4_mp4_preflight import probe_video_with_opencv

        video_probe = probe_video_with_opencv
    config.output_root.mkdir(parents=True, exist_ok=True)
    if config.expected_per_label <= 0:
        raise ValueError("expected_per_label must be positive")
    root_failures = _root_failures(config)
    if root_failures:
        summary = _base_summary(config, "invalid_roots")
        summary["failures"] = root_failures
        summary["records"] = []
        _write_outputs(config.output_root, summary)
        return summary
    if not config.staging_root.exists():
        summary = _base_summary(config, "missing_staging_root")
        summary.update(_empty_counts(config))
        summary["failures"] = [{"code": "missing_staging_root", "path": config.staging_root.as_posix()}]
        summary["records"] = []
        _write_outputs(config.output_root, summary)
        return summary

    heldout_hash_index = _heldout_hash_index(config.heldout_roots)
    records = _discover_records(config, heldout_hash_index, video_probe)
    failures = _record_failures(records)
    warnings = _record_warnings(records)
    valid_records = [record for record in records if str(record["status"]) == "valid"]
    counts = Counter(str(record["label"]) for record in valid_records)
    label_counts = {label: int(counts.get(label, 0)) for label in REVIEW_LABEL_ORDER}
    remaining_counts = {label: max(0, config.expected_per_label - label_counts[label]) for label in REVIEW_LABEL_ORDER}
    status = _status(
        records=records,
        failures=failures,
        label_counts=label_counts,
        expected_per_label=config.expected_per_label,
    )
    summary = {
        **_base_summary(config, status),
        "mp4_count": len(records),
        "valid_mp4_count": len(valid_records),
        "hashed_mp4_count": sum(1 for record in records if record.get("sha256")),
        "heldout_hash_count": len(heldout_hash_index),
        "failed_video_probe_count": sum(1 for record in records if record.get("video_failure_codes")),
        "label_counts": label_counts,
        "remaining_counts": remaining_counts,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "records": records,
        "audit_table": (config.output_root / "recording_staging_audit_table.csv").as_posix(),
    }
    _write_outputs(config.output_root, summary)
    return summary


def _base_summary(config: V4RecordingStagingAuditConfig, status: str) -> dict[str, object]:
    return {
        "status": status,
        "staging_root": config.staging_root.as_posix(),
        "calibration_root": config.calibration_root.as_posix() if config.calibration_root else None,
        "heldout_roots": [path.as_posix() for path in config.heldout_roots],
        "expected_per_label": int(config.expected_per_label),
        "expected_total": int(config.expected_per_label) * len(REVIEW_LABEL_ORDER),
    }


def _empty_counts(config: V4RecordingStagingAuditConfig) -> dict[str, object]:
    return {
        "mp4_count": 0,
        "valid_mp4_count": 0,
        "hashed_mp4_count": 0,
        "heldout_hash_count": 0,
        "failed_video_probe_count": 0,
        "label_counts": {label: 0 for label in REVIEW_LABEL_ORDER},
        "remaining_counts": {label: int(config.expected_per_label) for label in REVIEW_LABEL_ORDER},
        "failure_count": 1,
        "warning_count": 0,
        "warnings": [],
        "audit_table": (config.output_root / "recording_staging_audit_table.csv").as_posix(),
    }


def _root_failures(config: V4RecordingStagingAuditConfig) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    staging = _resolved(config.staging_root)
    if config.calibration_root is not None and _overlaps(staging, _resolved(config.calibration_root)):
        failures.append(
            {
                "code": "staging_overlaps_calibration_root",
                "staging_root": config.staging_root.as_posix(),
                "calibration_root": config.calibration_root.as_posix(),
            }
        )
    for heldout_root in config.heldout_roots:
        if _overlaps(staging, _resolved(heldout_root)):
            failures.append(
                {
                    "code": "staging_overlaps_heldout_root",
                    "staging_root": config.staging_root.as_posix(),
                    "heldout_root": heldout_root.as_posix(),
                }
            )
    return failures


def _discover_records(
    config: V4RecordingStagingAuditConfig,
    heldout_hash_index: dict[str, str],
    video_probe: VideoProbe,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(config.staging_root.rglob("*.mp4"), key=natural_key):
        label = _infer_label(path, config.staging_root)
        status = "valid"
        reasons: list[str] = []
        if label is None:
            status = "invalid"
            reasons.append("invalid_label_path")
        size_bytes = int(path.stat().st_size)
        sha256 = _sha256(path) if size_bytes > 0 else ""
        if size_bytes <= 0:
            status = "invalid"
            reasons.append("empty_mp4_file")
        if _is_heldout_path(path, config.heldout_roots):
            status = "invalid"
            reasons.append("heldout_path_leak")
        heldout_match_path = heldout_hash_index.get(sha256, "") if sha256 else ""
        if heldout_match_path:
            status = "invalid"
            reasons.append("heldout_content_hash_match")
        probe = _probe_if_nonempty(path, size_bytes, video_probe)
        video_failures = _video_failures(probe, config) if size_bytes > 0 else []
        if video_failures:
            status = "invalid"
            reasons.extend(str(failure["code"]) for failure in video_failures)
        records.append(
            {
                "status": status,
                "label": label or "",
                "path": path.as_posix(),
                "relative_path": path.relative_to(config.staging_root).as_posix(),
                "filename": path.name,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "heldout_match_path": heldout_match_path,
                "video_opened": probe.get("opened"),
                "width": probe.get("width"),
                "height": probe.get("height"),
                "frame_count": probe.get("frame_count"),
                "fps": probe.get("fps"),
                "duration_s": probe.get("duration_s"),
                "video_failure_codes": [str(failure["code"]) for failure in video_failures],
                "reasons": reasons,
                "warnings": [],
            }
        )
    _mark_duplicate_filenames(records)
    _mark_duplicate_content(records)
    return records


def _probe_if_nonempty(path: Path, size_bytes: int, video_probe: VideoProbe) -> dict[str, object]:
    if size_bytes <= 0:
        return {"opened": False, "width": 0, "height": 0, "frame_count": 0, "fps": 0.0, "duration_s": None}
    try:
        return video_probe(path)
    except Exception as exc:
        return {
            "opened": False,
            "width": 0,
            "height": 0,
            "frame_count": 0,
            "fps": 0.0,
            "duration_s": None,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }


def _video_failures(probe: dict[str, object], config: V4RecordingStagingAuditConfig) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    if not bool(probe.get("opened")):
        failures.append({"code": "video_not_opened", "reason": probe.get("failure_reason")})
    width = _int(probe.get("width"))
    height = _int(probe.get("height"))
    frame_count = _int(probe.get("frame_count"))
    fps = _float(probe.get("fps"))
    if width < config.min_width:
        failures.append({"code": "width_too_small", "actual": width, "minimum": config.min_width})
    if height < config.min_height:
        failures.append({"code": "height_too_small", "actual": height, "minimum": config.min_height})
    if frame_count < config.min_frame_count:
        failures.append({"code": "frame_count_too_small", "actual": frame_count, "minimum": config.min_frame_count})
    if fps < config.min_fps:
        failures.append({"code": "fps_too_small", "actual": fps, "minimum": config.min_fps})
    return failures


def _infer_label(path: Path, staging_root: Path) -> str | None:
    try:
        parts = path.relative_to(staging_root).parts
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    first = parts[0]
    return first if first in REVIEW_LABEL_ORDER else None


def _mark_duplicate_filenames(records: list[dict[str, object]]) -> None:
    counts = Counter(str(record["filename"]).lower() for record in records)
    for record in records:
        if counts[str(record["filename"]).lower()] <= 1:
            continue
        warnings = list(record.get("warnings", []))
        warnings.append("duplicate_filename")
        record["warnings"] = warnings


def _mark_duplicate_content(records: list[dict[str, object]]) -> None:
    counts = Counter(str(record.get("sha256", "")) for record in records if record.get("sha256"))
    for record in records:
        digest = str(record.get("sha256", ""))
        if not digest or counts[digest] <= 1:
            continue
        warnings = list(record.get("warnings", []))
        warnings.append("duplicate_content_hash")
        record["warnings"] = warnings


def _record_failures(records: list[dict[str, object]]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for record in records:
        if str(record["status"]) == "valid":
            continue
        failures.append(
            {
                "code": "invalid_staging_mp4",
                "path": record["path"],
                "relative_path": record["relative_path"],
                "reasons": record["reasons"],
                "heldout_match_path": record.get("heldout_match_path", ""),
            }
        )
    return failures


def _record_warnings(records: list[dict[str, object]]) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for record in records:
        record_warnings = list(record.get("warnings", []))
        if not record_warnings:
            continue
        warnings.append(
            {
                "code": "staging_mp4_warning",
                "path": record["path"],
                "relative_path": record["relative_path"],
                "warnings": record_warnings,
            }
        )
    return warnings


def _status(
    *,
    records: list[dict[str, object]],
    failures: list[dict[str, object]],
    label_counts: dict[str, int],
    expected_per_label: int,
) -> str:
    if failures:
        return "staging_needs_review"
    if not records:
        return "awaiting_staging_mp4s"
    if all(label_counts[label] >= expected_per_label for label in REVIEW_LABEL_ORDER):
        return "staging_ready_for_assignment"
    return "partial_staging_ready"


def _is_heldout_path(path: Path, heldout_roots: tuple[Path, ...]) -> bool:
    resolved = _resolved(path)
    return any(_is_within(resolved, _resolved(heldout_root)) for heldout_root in heldout_roots)


def _heldout_hash_index(heldout_roots: tuple[Path, ...]) -> dict[str, str]:
    index: dict[str, str] = {}
    for heldout_root in heldout_roots:
        if not heldout_root.exists():
            continue
        for path in sorted(heldout_root.rglob("*.mp4"), key=natural_key):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            digest = _sha256(path)
            index.setdefault(digest, path.as_posix())
    return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_outputs(output_root: Path, summary: dict[str, object]) -> None:
    (output_root / "recording_staging_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_table(output_root / "recording_staging_audit_table.csv", list(summary.get("records", [])))
    (output_root / "recording_staging_audit_summary.md").write_text(_markdown(summary), encoding="utf-8")


def _write_table(path: Path, records: list[object]) -> None:
    fieldnames = [
        "status",
        "label",
        "relative_path",
        "filename",
        "size_bytes",
        "sha256",
        "heldout_match_path",
        "video_opened",
        "width",
        "height",
        "frame_count",
        "fps",
        "duration_s",
        "video_failure_codes",
        "reasons",
        "warnings",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if not isinstance(record, dict):
                continue
            row = {field: record.get(field) for field in fieldnames}
            if isinstance(row["video_failure_codes"], list):
                row["video_failure_codes"] = ";".join(str(item) for item in row["video_failure_codes"])
            row["reasons"] = ";".join(str(reason) for reason in record.get("reasons", []))
            row["warnings"] = ";".join(str(warning) for warning in record.get("warnings", []))
            writer.writerow(row)


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# V4 Recording Staging Audit",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Staging root: `{summary.get('staging_root')}`",
        f"- Expected per label: `{summary.get('expected_per_label')}`",
        f"- MP4 count: `{summary.get('mp4_count', 0)}`",
        f"- Valid MP4 count: `{summary.get('valid_mp4_count', 0)}`",
        f"- Hashed MP4 count: `{summary.get('hashed_mp4_count', 0)}`",
        f"- Held-out hash count: `{summary.get('heldout_hash_count', 0)}`",
        f"- Failed video probe count: `{summary.get('failed_video_probe_count', 0)}`",
        f"- Failure count: `{summary.get('failure_count', 0)}`",
        f"- Warning count: `{summary.get('warning_count', 0)}`",
        "",
        "## Label Counts",
        "",
        "| Label | Valid MP4s | Remaining |",
        "|---|---:|---:|",
    ]
    label_counts = summary.get("label_counts")
    remaining_counts = summary.get("remaining_counts")
    if isinstance(label_counts, dict) and isinstance(remaining_counts, dict):
        for label in REVIEW_LABEL_ORDER:
            lines.append(f"| `{label}` | `{label_counts.get(label, 0)}` | `{remaining_counts.get(label, 0)}` |")
    failures = summary.get("failures")
    if isinstance(failures, list) and failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:20]:
            if isinstance(failure, dict):
                lines.append(f"- `{failure.get('code')}`: `{failure.get('relative_path', failure.get('path', ''))}`")
    lines.extend(["", "## Next Step", "", _next_step(str(summary.get("status"))), ""])
    return "\n".join(lines)


def _next_step(status: str) -> str:
    if status == "staging_ready_for_assignment":
        return "Run the staging-to-slot assignment dry run and review the mapping table."
    if status == "partial_staging_ready":
        return "Add more MP4s for labels with remaining counts, or intentionally proceed with a partial assignment."
    if status == "awaiting_staging_mp4s":
        return "Record or add MP4s under `v4_recording_staging/{rock,paper,scissors}/`."
    if status == "missing_staging_root":
        return "Create the staging scaffold before recording or adding MP4s."
    return "Fix the reported staging audit failures before continuing."


def _int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["V4RecordingStagingAuditConfig", "VideoProbe", "audit_v4_recording_staging"]
