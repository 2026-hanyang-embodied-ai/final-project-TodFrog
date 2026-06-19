"""Executor for prepared v4 skeleton-review plans."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from embodied_rps.real_skeleton_review import discover_skeleton_review_videos, validate_skeleton_review_discovery

ReviewRunner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class V4SkeletonReviewExecutionConfig:
    """Configuration for executing a prepared v4 skeleton-review plan."""

    skeleton_review_plan_path: Path
    output_root: Path
    dry_run: bool = False


def run_v4_skeleton_review_from_plan(
    config: V4SkeletonReviewExecutionConfig,
    *,
    review_runner: ReviewRunner | None = None,
) -> dict[str, object]:
    """Execute a prepared v4 MediaPipe skeleton-review plan when its input is ready."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    if not config.skeleton_review_plan_path.exists():
        summary = {
            "status": "awaiting_skeleton_review_plan",
            "next_action": "run_v4_pipeline_or_prepare_v4_skeleton_review_plan",
            "blocking_stage": "skeleton_review_plan",
            "skeleton_review_plan": config.skeleton_review_plan_path.as_posix(),
            "execution_output_root": config.output_root.as_posix(),
            "dry_run": config.dry_run,
            "input_root": "",
            "review_output_root": "",
            "expected_count": None,
            "expected_per_label": None,
            "review_command": [],
            "review_command_text": "",
            "discovery": {},
            "skipped_reason": "skeleton_review_plan_missing",
            "runner_exit_code": None,
            "review_manifest": None,
            "review_manifest_status": None,
            "visual_approval_required": False,
            "notes": [
                "This executor creates MediaPipe skeleton review artifacts only.",
                "It does not build the v4 calibration seed package.",
                "It does not train a model or launch SCHUNK/Isaac rendering.",
            ],
        }
        _write_summary(config.output_root, summary)
        return summary

    plan = _load_json(config.skeleton_review_plan_path)
    command = _review_command(plan)
    args = _extract_module_args(command)
    output_root = Path(str(plan.get("review_output_root", "")))
    manifest_path = output_root / "manifest.json"

    if plan.get("status") != "ready_for_skeleton_review":
        summary = _summary(
            config=config,
            plan=plan,
            command=command,
            status="awaiting_ready_review_plan",
            next_action="record_or_add_v4_calibration_videos",
            blocking_stage="skeleton_review_plan",
            skipped_reason="plan_status_not_ready_for_skeleton_review",
        )
        _write_summary(config.output_root, summary)
        return summary

    try:
        videos = discover_skeleton_review_videos(
            Path(str(plan.get("input_root", ""))),
            output_prefix=str(_option_value(args, "--output-prefix") or "v4_calibration"),
        )
        discovery = validate_skeleton_review_discovery(
            videos,
            expected_count=int(_number(plan.get("expected_count"), 0)),
            expected_per_label=int(_number(plan.get("expected_per_label"), 0)),
        )
    except (FileNotFoundError, ValueError) as exc:
        summary = _summary(
            config=config,
            plan=plan,
            command=command,
            status="input_not_ready_for_skeleton_review",
            next_action="fix_v4_calibration_input",
            blocking_stage="skeleton_review_input",
            skipped_reason=str(exc),
        )
        _write_summary(config.output_root, summary)
        return summary

    if not bool(discovery.get("passed")):
        summary = _summary(
            config=config,
            plan=plan,
            command=command,
            status="input_not_ready_for_skeleton_review",
            next_action="fix_v4_calibration_input",
            blocking_stage="skeleton_review_input",
            discovery=discovery,
            skipped_reason="discovery_validation_failed",
        )
        _write_summary(config.output_root, summary)
        return summary

    if config.dry_run:
        summary = _summary(
            config=config,
            plan=plan,
            command=command,
            status="ready_for_review_execution",
            next_action="run_without_dry_run_to_create_review_artifacts",
            blocking_stage="manual_review_execution",
            discovery=discovery,
        )
        _write_summary(config.output_root, summary)
        return summary

    runner = review_runner or _default_review_runner
    exit_code = int(runner(args))
    manifest: Mapping[str, object] = _load_json(manifest_path) if manifest_path.exists() else {}
    if exit_code != 0:
        status = "review_execution_failed"
        next_action = "inspect_review_executor_error"
        blocking_stage = "skeleton_review_execution"
    elif not manifest_path.exists():
        status = "review_manifest_missing"
        next_action = "rerun_skeleton_review_executor"
        blocking_stage = "skeleton_review_manifest"
    elif manifest.get("status") == "passed":
        status = "skeleton_review_passed"
        next_action = "visually_review_skeleton_outputs_then_build_seed_package"
        blocking_stage = "visual_skeleton_approval"
    else:
        status = "skeleton_review_failed"
        next_action = "inspect_skeleton_review_failures"
        blocking_stage = "skeleton_review_quality"

    summary = _summary(
        config=config,
        plan=plan,
        command=command,
        status=status,
        next_action=next_action,
        blocking_stage=blocking_stage,
        discovery=discovery,
        runner_exit_code=exit_code,
        review_manifest=manifest_path.as_posix(),
        review_manifest_status=manifest.get("status"),
    )
    _write_summary(config.output_root, summary)
    return summary


def _default_review_runner(args: Sequence[str]) -> int:
    from embodied_rps.tools.extract_real_hand_skeleton_review import main

    return int(main(list(args)))


def _summary(
    *,
    config: V4SkeletonReviewExecutionConfig,
    plan: Mapping[str, object],
    command: Sequence[str],
    status: str,
    next_action: str,
    blocking_stage: str,
    discovery: Mapping[str, object] | None = None,
    skipped_reason: str | None = None,
    runner_exit_code: int | None = None,
    review_manifest: str | None = None,
    review_manifest_status: object | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "next_action": next_action,
        "blocking_stage": blocking_stage,
        "skeleton_review_plan": config.skeleton_review_plan_path.as_posix(),
        "execution_output_root": config.output_root.as_posix(),
        "dry_run": config.dry_run,
        "input_root": str(plan.get("input_root", "")),
        "review_output_root": str(plan.get("review_output_root", "")),
        "expected_count": plan.get("expected_count"),
        "expected_per_label": plan.get("expected_per_label"),
        "review_command": list(command),
        "review_command_text": _quote_command(command),
        "discovery": dict(discovery or {}),
        "skipped_reason": skipped_reason,
        "runner_exit_code": runner_exit_code,
        "review_manifest": review_manifest,
        "review_manifest_status": review_manifest_status,
        "visual_approval_required": status == "skeleton_review_passed",
        "notes": [
            "This executor creates MediaPipe skeleton review artifacts only.",
            "It does not build the v4 calibration seed package.",
            "It does not train a model or launch SCHUNK/Isaac rendering.",
        ],
    }


def _write_summary(output_root: Path, summary: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "skeleton_review_execution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "skeleton_review_execution_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")


def _summary_markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# V4 Skeleton Review Execution Summary",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Next action: `{summary.get('next_action')}`",
        f"- Blocking stage: `{summary.get('blocking_stage')}`",
        f"- Input root: `{summary.get('input_root')}`",
        f"- Review output root: `{summary.get('review_output_root')}`",
        f"- Dry run: `{summary.get('dry_run')}`",
        "",
        "## Discovery",
        "",
    ]
    discovery = _mapping(summary.get("discovery"))
    if discovery:
        lines.append(f"- Passed: `{discovery.get('passed')}`")
        lines.append(f"- Video count: `{discovery.get('video_count')}`")
        lines.append(f"- Label counts: `{json.dumps(discovery.get('label_counts', {}), ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- Duplicate count: `{discovery.get('duplicate_count')}`")
    else:
        lines.append("- Not run.")
    if summary.get("skipped_reason"):
        lines.extend(["", "## Skipped Reason", "", f"- `{summary.get('skipped_reason')}`"])
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```powershell",
            str(summary.get("review_command_text", "")),
            "```",
            "",
            "## Notes",
            "",
        ]
    )
    notes = summary.get("notes")
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _review_command(plan: Mapping[str, object]) -> list[str]:
    command = plan.get("review_command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        return [str(part) for part in command]
    text = plan.get("review_command_text")
    if isinstance(text, str) and text.strip():
        return [text]
    return []


def _extract_module_args(command: Sequence[str]) -> list[str]:
    parts = list(command)
    if len(parts) >= 3 and parts[1] == "-m":
        return parts[3:]
    if parts and parts[0].endswith("extract_real_hand_skeleton_review"):
        return parts[1:]
    return parts


def _option_value(args: Sequence[str], option: str) -> str | None:
    for index, value in enumerate(args):
        if value == option and index + 1 < len(args):
            return str(args[index + 1])
    return None


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quote_command(parts: Sequence[str]) -> str:
    quoted: list[str] = []
    for part in parts:
        if not part or any(char.isspace() for char in part):
            quoted.append(f"'{part}'")
        else:
            quoted.append(part)
    return " ".join(quoted)


__all__ = ["V4SkeletonReviewExecutionConfig", "run_v4_skeleton_review_from_plan"]
