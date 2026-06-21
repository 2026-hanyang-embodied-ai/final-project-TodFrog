"""Build final-submission documentation and report support artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-untyped]


DatasetStatus = Literal["upload_ready", "metadata_only", "missing"]
DatasetRole = Literal["primary", "supporting", "diagnostic", "metadata_only"]

FINAL_LIVE_VIDEO = Path(
    "artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/"
    "selected_final_submission_take/final_submission_live_counterattack.mp4"
)
FINAL_LIVE_POSTER = Path(
    "artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/"
    "selected_final_submission_take/poster_frame.png"
)
FINAL_LIVE_SUMMARY = Path(
    "artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/"
    "selected_final_submission_take/selected_take_summary.json"
)
THREE_TAKE_REVIEW_VIDEO = Path(
    "artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/"
    "combined_three_take_review/final_submission_three_take_counterattack_compilation.mp4"
)
THREE_TAKE_REVIEW_POSTER = Path(
    "artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/"
    "combined_three_take_review/three_take_compilation_poster.png"
)
FINAL_V4_CONFIG = Path("configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml")
SCHUNK_STYLE_ROOT = Path("artifacts/schunk_joint_target_skeleton_passed")
CANONICAL_SEED_PREVIEW = Path("artifacts/real_skeleton_canonical_20260610/preview/rock_to_paper_01_canonical_preview.png")
REAL_GUIDED_AUGMENTATION_PREVIEW = Path(
    "artifacts/real_skeleton_bulk_aligned_20260610/preview/bulk_augmented_samples_preview.png"
)
REAL_GUIDED_ALIGNMENT_ERROR = Path(
    "artifacts/real_guided_render_ready_bulk_alignment_20260610/preview/progress_error_comparison.png"
)
REAL_GUIDED_POSE_SCHEDULE = Path(
    "artifacts/real_guided_render_ready_bulk_alignment_20260610/preview/real_guided_pose_schedule.png"
)
REPORT_TEMPLATE_ROOT = Path("conference-latex-template/IEEE-conference-template-062824")
DOCS_ROOT = Path("docs")
NOTEBOOK_PATH = Path("final-project.ipynb")

FORBIDDEN_FINAL_VIDEO_TOKENS = (
    "artifacts/realtime_demo_rehearsal_20260616",
    "artifacts/final_submission_prompt_counterattack_demo_20260619/replay_rehearsal",
)
PROTECTED_PDFS = (Path("proposal.pdf"), Path("presentation-slides.pdf"))


@dataclass(frozen=True)
class DatasetCandidate:
    """One local dataset artifact that the user may upload to Google Drive."""

    name: str
    path: Path
    role: DatasetRole
    description: str
    required_for: str
    drive_placeholder: str = "TODO: paste Google Drive URL after upload"


@dataclass(frozen=True)
class DatasetInventoryRow:
    """Resolved dataset row written to markdown and JSON."""

    name: str
    path: str
    role: DatasetRole
    status: DatasetStatus
    file_count: int
    size_bytes: int
    description: str
    required_for: str
    drive_placeholder: str


def default_dataset_candidates() -> tuple[DatasetCandidate, ...]:
    """Return final-submission dataset upload candidates."""

    return (
        DatasetCandidate(
            name="compact_real_guided_training_package",
            path=Path("artifacts/real_guided_training_package_20260610"),
            role="primary",
            description="Compact split-aware real-guided skeleton training package.",
            required_for="small reproducibility package and dataset schema reference",
        ),
        DatasetCandidate(
            name="large_real_guided_sharded_dataset",
            path=Path("artifacts/real_guided_large_sharded_20260610"),
            role="supporting",
            description="Large real-skeleton-guided sharded dataset used as a base for later expansions.",
            required_for="large-scale skeleton sequence experiments",
        ),
        DatasetCandidate(
            name="v4_final_gate_micro_dataset",
            path=Path("artifacts/real_guided_three_class_wait_expanded_v4_final_gate_micro_g1000_20260616"),
            role="supporting",
            description="Upload-ready v4 final-gate micro dataset currently present with shards.",
            required_for="v4 diagnostic/final-gate profile provenance",
        ),
        DatasetCandidate(
            name="v7e_three_class_diagnostic_dataset",
            path=Path("artifacts/real_guided_three_class_wait_expanded_v7e_stage1_paper_transition_rescue_20260619"),
            role="diagnostic",
            description="Full-scale v7e diagnostic three-class dataset; not promoted to live/demo.",
            required_for="report diagnostics only",
        ),
        DatasetCandidate(
            name="v7e_stage1_rock_transition_diagnostic_dataset",
            path=Path("artifacts/real_guided_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_20260619"),
            role="diagnostic",
            description="Full-scale v7e stage1 diagnostic remap; not promoted to live/demo.",
            required_for="report diagnostics only",
        ),
        DatasetCandidate(
            name="v4_fewshot_aug_dataset_root",
            path=Path("artifacts/real_guided_three_class_wait_expanded_v4_fewshot_aug_20260615"),
            role="metadata_only",
            description="Profile metadata exists in the current root, but shard files are absent.",
            required_for="v4 live profile provenance if restored from archive",
        ),
        DatasetCandidate(
            name="v4_rebalanced_dataset_root",
            path=Path("artifacts/real_guided_three_class_wait_expanded_v4_rebalanced_20260615"),
            role="metadata_only",
            description="Profile metadata exists in the current root, but shard files are absent.",
            required_for="v4 live profile provenance if restored from archive",
        ),
    )


def build_final_submission_assets(
    *,
    project_root: Path,
    docs_root: Path = DOCS_ROOT,
    report_root: Path = REPORT_TEMPLATE_ROOT,
    tcn_image: Path | None = None,
) -> dict[str, object]:
    """Build docs, report figures, dataset inventory, manifest, and notebook."""

    project_root = project_root.resolve(strict=False)
    docs_root = project_root / docs_root
    report_root = project_root / report_root
    figure_root = report_root / "figures"
    docs_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    _validate_final_video_paths(project_root)
    figure_outputs = copy_verified_report_figures(
        project_root=project_root,
        figure_root=figure_root,
        tcn_image=tcn_image,
    )
    dataset_rows = resolve_dataset_inventory(project_root=project_root)
    dataset_outputs = write_dataset_inventory(
        rows=dataset_rows,
        docs_root=docs_root,
        project_root=project_root,
    )
    notebook_path = write_final_project_notebook(project_root=project_root)
    manifest = write_submission_manifest(
        project_root=project_root,
        docs_root=docs_root,
        figure_outputs=figure_outputs,
        dataset_rows=dataset_rows,
        notebook_path=notebook_path,
    )
    scan = scan_generated_text_for_forbidden_content(
        project_root=project_root,
        roots=[docs_root, report_root / "final_report.tex", report_root / "final_report.bib", notebook_path],
    )
    payload = {
        "status": "passed",
        "claim_scope": "submission documentation and report support assets; no retraining and no protected PDF edits",
        "outputs": {
            "dataset_inventory_md": dataset_outputs["markdown"],
            "dataset_inventory_json": dataset_outputs["json"],
            "submission_manifest_json": manifest["path"],
            "notebook": _relative_path(notebook_path, project_root=project_root),
            "report_figure_dir": _relative_path(figure_root, project_root=project_root),
        },
        "figures": figure_outputs,
        "dataset_inventory": [row.__dict__ for row in dataset_rows],
        "forbidden_content_scan": scan,
    }
    _write_json(docs_root / "submission_assets_summary.json", payload)
    return payload


def copy_verified_report_figures(
    *,
    project_root: Path,
    figure_root: Path,
    tcn_image: Path | None = None,
) -> dict[str, str]:
    """Copy verified figures and create a labeled system schematic."""

    figure_root.mkdir(parents=True, exist_ok=True)
    copy_plan = {
        "final_live_candidate_poster": FINAL_LIVE_POSTER,
        "three_take_review_poster": THREE_TAKE_REVIEW_POSTER,
        "canonical_skeleton_seed": CANONICAL_SEED_PREVIEW,
        "real_guided_skeleton_augmentation": REAL_GUIDED_AUGMENTATION_PREVIEW,
        "real_guided_alignment_error": REAL_GUIDED_ALIGNMENT_ERROR,
        "real_guided_pose_schedule": REAL_GUIDED_POSE_SCHEDULE,
        "schunk_rock_yaw45_pitch20": SCHUNK_STYLE_ROOT / "rock_view_yaw45_pitch20.png",
        "schunk_paper_yaw45_pitch20": SCHUNK_STYLE_ROOT / "paper_view_yaw45_pitch20.png",
        "schunk_scissors_yaw45_pitch20": SCHUNK_STYLE_ROOT / "scissors_view_yaw45_pitch20.png",
    }
    outputs: dict[str, str] = {}
    for label, source in copy_plan.items():
        source_path = project_root / source
        if not source_path.exists():
            raise FileNotFoundError(f"Missing verified report figure source: {source}")
        target = figure_root / f"{label}.png"
        shutil.copyfile(source_path, target)
        outputs[label] = _relative_path(target, project_root=project_root)

    if tcn_image is not None:
        if not tcn_image.exists():
            raise FileNotFoundError(f"Missing provided TCN image: {tcn_image}")
        tcn_target = figure_root / "tcn_temporal_model.png"
        shutil.copyfile(tcn_image, tcn_target)
        outputs["tcn_temporal_model"] = _relative_path(tcn_target, project_root=project_root)

    system_diagram = figure_root / "verified_system_pipeline.png"
    _write_system_pipeline_diagram(system_diagram)
    outputs["verified_system_pipeline"] = _relative_path(system_diagram, project_root=project_root)
    return outputs


def resolve_dataset_inventory(*, project_root: Path) -> list[DatasetInventoryRow]:
    """Resolve dataset candidates into upload status rows."""

    rows: list[DatasetInventoryRow] = []
    for candidate in default_dataset_candidates():
        absolute = project_root / candidate.path
        if not absolute.exists():
            status: DatasetStatus = "missing"
            file_count = 0
            size_bytes = 0
        else:
            files = [path for path in absolute.rglob("*") if path.is_file()]
            file_count = len(files)
            size_bytes = sum(path.stat().st_size for path in files)
            has_shards = any(path.suffix == ".npz" and "shards" in path.parts for path in files)
            status = "metadata_only" if candidate.role == "metadata_only" or not has_shards else "upload_ready"
        rows.append(
            DatasetInventoryRow(
                name=candidate.name,
                path=candidate.path.as_posix(),
                role=candidate.role,
                status=status,
                file_count=file_count,
                size_bytes=size_bytes,
                description=candidate.description,
                required_for=candidate.required_for,
                drive_placeholder=candidate.drive_placeholder,
            )
        )
    return rows


def write_dataset_inventory(
    *,
    rows: list[DatasetInventoryRow],
    docs_root: Path,
    project_root: Path,
) -> dict[str, str]:
    """Write dataset inventory markdown and JSON for the README handoff."""

    markdown_path = docs_root / "dataset_inventory.md"
    json_path = docs_root / "dataset_inventory.json"
    lines = [
        "# Dataset Inventory For Google Drive Upload",
        "",
        "All paths are workspace-relative. Large datasets should be uploaded to Google Drive and linked in the README.",
        "Heldout `*/test` MP4s remain validation-only and are not training or demo-generation inputs.",
        "",
        "| Name | Status | Role | Local path | Size | Google Drive URL |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | `{status}` | `{role}` | `{path}` | {size} | {url} |".format(
                name=row.name,
                status=row.status,
                role=row.role,
                path=row.path,
                size=_format_bytes(row.size_bytes),
                url=row.drive_placeholder,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `upload_ready` means shard files are present in this workspace root.",
            "- `metadata_only` means summaries/profile references exist, but shard files are not present here.",
            "- v7e datasets are diagnostics only; they are not promoted to the live/demo predictor.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(json_path, {"datasets": [row.__dict__ for row in rows]})
    return {
        "markdown": _relative_path(markdown_path, project_root=project_root),
        "json": _relative_path(json_path, project_root=project_root),
    }


def write_submission_manifest(
    *,
    project_root: Path,
    docs_root: Path,
    figure_outputs: dict[str, str],
    dataset_rows: list[DatasetInventoryRow],
    notebook_path: Path,
) -> dict[str, str]:
    """Write a machine-readable submission manifest."""

    manifest_path = docs_root / "submission_manifest.json"
    selected_summary = _read_json(project_root / FINAL_LIVE_SUMMARY)
    payload = {
        "status": "passed",
        "repository": "https://github.com/2026-hanyang-embodied-ai/final-project-TodFrog",
        "project_title": "Actuator-Constrained Early Intention Prediction for Simulated Rock-Paper-Scissors Robot Hands",
        "research_focus": (
            "few-shot real MediaPipe skeleton capture -> real-guided skeleton "
            "augmentation/alignment -> sim-to-real live counterattack validation"
        ),
        "pipeline_evidence": {
            "real_clips": 20,
            "processed_frames": 720,
            "canonical_sequences": 20,
            "feature_dimension": 142,
            "compact_augmented_samples": 2000,
            "large_sharded_samples": 10000,
            "large_split": [7000, 1500, 1500],
            "alignment_frames_per_transition": 32,
            "aligned_manifest_entries": 64,
        },
        "final_live_candidate": {
            "video": FINAL_LIVE_VIDEO.as_posix(),
            "poster": FINAL_LIVE_POSTER.as_posix(),
            "selected_take": selected_summary.get("selected_take_id"),
            "episode_result": selected_summary.get("episode_result"),
        },
        "supporting_review_video": THREE_TAKE_REVIEW_VIDEO.as_posix(),
        "live_demo_policy": {
            "family": "v4",
            "config": FINAL_V4_CONFIG.as_posix(),
            "v7e_policy": "diagnostics only; not promoted",
            "v7f_policy": "not started",
        },
        "report": {
            "tex": "conference-latex-template/IEEE-conference-template-062824/final_report.tex",
            "pdf": "conference-latex-template/IEEE-conference-template-062824/final_report.pdf",
            "figures": figure_outputs,
        },
        "notebook": _relative_path(notebook_path, project_root=project_root),
        "datasets": [row.__dict__ for row in dataset_rows],
        "links_to_fill_after_upload": {
            "dataset_google_drive": "TODO",
            "demo_video_youtube": "TODO",
            "presentation_video_youtube": "TODO",
        },
        "protected_pdf_policy": {
            "proposal.pdf": "not edited",
            "presentation-slides.pdf": "not edited",
        },
    }
    _write_json(manifest_path, payload)
    return {"path": _relative_path(manifest_path, project_root=project_root)}


def write_final_project_notebook(*, project_root: Path) -> Path:
    """Write a lightweight executed-style notebook for project reproduction."""

    notebook_path = project_root / NOTEBOOK_PATH
    notebook = {
        "cells": [
            _markdown_cell(
                "# Final Project: Actuator-Constrained Early Intention Prediction\n\n"
                "This notebook summarizes the verified real-to-sim-to-real skeleton pipeline, final submission artifacts, and validation commands.",
                cell_id="overview",
            ),
            _markdown_cell(
                "## Dataset Pipeline\n\n"
                "- Real seed capture: 20 MediaPipe-reviewed clips and 720 processed frames\n"
                "- Canonical representation: 20 sequences with 142-dimensional per-frame features\n"
                "- Real-guided expansion: 2,000 compact samples and 10,000 large sharded samples\n"
                "- Sim-to-real validation: frozen v4 predictor applied back to prompt-window live capture.",
                cell_id="dataset-pipeline",
            ),
            _markdown_cell(
                "## Final Demo Policy\n\n"
                "- Live/demo predictor: v4 fallback policy\n"
                "- Config: `configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml`\n"
                "- v7e is diagnostics-only and is not promoted.",
                cell_id="final-demo-policy",
            ),
            _code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "root = Path.cwd()\n"
                "manifest = json.loads((root / 'docs/submission_manifest.json').read_text(encoding='utf-8'))\n"
                "manifest['final_live_candidate']",
                execution_count=1,
                cell_id="manifest-final-candidate",
            ),
            _markdown_cell(
                "## Dataset Handoff\n\n"
                "The dataset files are intentionally not committed to GitHub. Upload the `upload_ready` rows from `docs/dataset_inventory.md` to Google Drive and paste the links into the README.",
                cell_id="dataset-handoff",
            ),
            _code_cell(
                "import json\n"
                "inventory = json.loads(Path('docs/dataset_inventory.json').read_text(encoding='utf-8'))\n"
                "[(row['name'], row['status'], row['path']) for row in inventory['datasets']]",
                execution_count=2,
                cell_id="dataset-inventory",
            ),
            _markdown_cell(
                "## Re-run Focused Validation\n\n"
                "Use the command below from the repository root after installing dependencies.",
                cell_id="validation-command-intro",
            ),
            _code_cell(
                "print('PYTHONPATH=src python -m pytest tests/test_current_best_realtime_demo_cli.py tests/test_final_robot_counterattack_demo.py tests/test_final_submission_live_counterattack_recording.py tests/test_policy.py tests/test_feasibility.py tests/test_final_submission_assets.py -q')",
                execution_count=3,
                cell_id="validation-command",
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _write_json(notebook_path, notebook)
    return notebook_path


def scan_generated_text_for_forbidden_content(*, project_root: Path, roots: list[Path]) -> dict[str, object]:
    """Fail if generated durable text contains local absolute paths or bad final claims."""

    scanned: list[str] = []
    violations: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            if path.suffix.lower() not in {".bib", ".csv", ".json", ".md", ".tex", ".txt", ".ipynb"}:
                continue
            text = path.read_text(encoding="utf-8")
            rel = _relative_path(path, project_root=project_root)
            scanned.append(rel)
            for token in ("C:\\", "C:/", "Users\\user", "Users/user"):
                if token in text:
                    violations.append({"path": rel, "token": token})
            for token in FORBIDDEN_FINAL_VIDEO_TOKENS:
                if token in text and "validation" not in text.lower():
                    violations.append({"path": rel, "token": token})
    if violations:
        raise ValueError(f"Forbidden generated text content: {violations}")
    return {"status": "passed", "scanned": scanned, "violation_count": 0}


def validate_no_heldout_test_mp4(path: Path, *, usage: Literal["training", "demo", "validation"]) -> None:
    """Reject heldout test MP4s outside validation-only usage."""

    normalized = path.as_posix().lower()
    if usage != "validation" and "/test/" in normalized and normalized.endswith(".mp4"):
        raise ValueError(f"Heldout test MP4s are validation-only: {path.as_posix()}")


def _validate_final_video_paths(project_root: Path) -> None:
    expected_review_video = project_root / THREE_TAKE_REVIEW_VIDEO
    root_review_video = project_root / "final_submission_three_take_counterattack_compilation.mp4"
    if not expected_review_video.exists() and root_review_video.exists():
        expected_review_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root_review_video, expected_review_video)

    final_video = project_root / FINAL_LIVE_VIDEO
    review_video = expected_review_video
    for path in (final_video, review_video):
        if not path.exists():
            raise FileNotFoundError(f"Missing required final submission video artifact: {_relative_path(path, project_root=project_root)}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Video artifact is empty: {_relative_path(path, project_root=project_root)}")
    final_text = FINAL_LIVE_VIDEO.as_posix()
    if any(token in final_text for token in FORBIDDEN_FINAL_VIDEO_TOKENS):
        raise ValueError("Final submission video must not point to replay rehearsal artifacts")


def _write_system_pipeline_diagram(path: Path) -> None:
    width, height = 1600, 420
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = _load_diagram_font(30)
    label_font = _load_diagram_font(24)
    footer_font = _load_diagram_font(18)
    title = "Real-to-sim-to-real skeleton pipeline"
    draw.text((40, 28), title, fill=(15, 23, 42), font=title_font)
    boxes = [
        ("Few real\nRPS videos", (40, 120, 245, 285), (219, 234, 254)),
        ("MediaPipe\n21 landmarks", (285, 120, 490, 285), (220, 252, 231)),
        ("Canonical\nskeleton seeds", (530, 120, 735, 285), (254, 249, 195)),
        ("Real-guided\naugmentation", (775, 120, 980, 285), (255, 237, 213)),
        ("TCN predictor\nsim-to-real", (1020, 120, 1225, 285), (237, 233, 254)),
        ("Actuator-feasible\nrobot response", (1265, 120, 1530, 285), (229, 231, 235)),
    ]
    for index, (label, box, color) in enumerate(boxes):
        draw.rounded_rectangle(box, radius=12, fill=color, outline=(71, 85, 105), width=2)
        _center_multiline(draw, label, box, font=label_font)
        if index < len(boxes) - 1:
            x2 = box[2]
            next_x1 = boxes[index + 1][1][0]
            y = (box[1] + box[3]) // 2
            draw.line((x2 + 12, y, next_x1 - 12, y), fill=(30, 41, 59), width=3)
            draw.polygon([(next_x1 - 12, y), (next_x1 - 28, y - 8), (next_x1 - 28, y + 8)], fill=(30, 41, 59))
    draw.text(
        (40, 350),
        "Schematic built from verified project artifacts; not a fabricated fresh Isaac Sim screenshot.",
        fill=(71, 85, 105),
        font=footer_font,
    )
    image.save(path)


def _center_multiline(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], *, font: ImageFont.ImageFont) -> None:
    lines = text.splitlines()
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_h = (bbox[3] - bbox[1]) + 8
    total_h = line_h * len(lines)
    y = box[1] + ((box[3] - box[1]) - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = box[0] + ((box[2] - box[0]) - line_w) // 2
        draw.text((x, y), line, fill=(15, 23, 42), font=font)
        y += line_h


def _load_diagram_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _markdown_cell(source: str, *, cell_id: str) -> dict[str, object]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str, *, execution_count: int, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(project_root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} TB"


def write_metrics_csv(path: Path) -> None:
    """Write compact final-report metrics table used by README/report."""

    rows = [
        ("real seed capture", "MediaPipe-reviewed clips", "20", "720 processed frames; minimum detection coverage 1.0"),
        ("canonical seed dataset", "feature dimension", "142", "20 sequences; max sequence length 56"),
        ("real-guided augmentation", "compact samples", "2000", "1000 rock_to_paper and 1000 rock_to_scissors"),
        ("real-guided augmentation", "large sharded samples", "10000", "split 7000/1500/1500"),
        ("render-ready alignment", "max progress error", "0.016", "32-frame real-guided schedule; old 8-frame max about 0.106"),
        ("v4 fallback", "profile weights", "[0.25, 0.75, 0.0]", "fewshot_aug_tcn, rebalanced_tcn, final_gate_micro"),
        ("v4 fallback", "original20 strict validation", "20/20", "from v4 two-stage selector policy source"),
        ("v4 fallback", "heldout15 validation", "11/15", "validation-only; not used for training/demo generation"),
        ("v7e diagnostics", "original20 strict validation", "17/20", "diagnostic only; not promoted"),
        ("final live take", "selected take", "human rock -> robot paper", "prompt-scissors retake"),
        ("final live take", "episode result", "actuator_feasible_win", "remaining actuator time 0.4667 s"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value", "notes"])
        writer.writerows(rows)


__all__ = [
    "DatasetCandidate",
    "DatasetInventoryRow",
    "FINAL_LIVE_VIDEO",
    "THREE_TAKE_REVIEW_VIDEO",
    "build_final_submission_assets",
    "copy_verified_report_figures",
    "default_dataset_candidates",
    "resolve_dataset_inventory",
    "scan_generated_text_for_forbidden_content",
    "validate_no_heldout_test_mp4",
    "write_dataset_inventory",
    "write_final_project_notebook",
    "write_metrics_csv",
    "write_submission_manifest",
]
