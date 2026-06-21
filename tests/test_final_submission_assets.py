from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from embodied_rps.submission_assets import (
    copy_verified_report_figures,
    resolve_dataset_inventory,
    scan_generated_text_for_forbidden_content,
    validate_no_heldout_test_mp4,
    write_dataset_inventory,
    write_final_project_notebook,
    write_submission_manifest,
)


def test_dataset_inventory_marks_missing_shards_as_metadata_only() -> None:
    rows = resolve_dataset_inventory(project_root=Path.cwd())
    by_name = {row.name: row for row in rows}

    assert by_name["compact_real_guided_training_package"].status == "upload_ready"
    assert by_name["large_real_guided_sharded_dataset"].status == "upload_ready"
    assert by_name["v4_final_gate_micro_dataset"].status == "upload_ready"
    assert by_name["v4_fewshot_aug_dataset_root"].status == "metadata_only"
    assert by_name["v4_rebalanced_dataset_root"].status == "metadata_only"


def test_write_dataset_inventory_uses_workspace_relative_paths(tmp_path: Path) -> None:
    rows = resolve_dataset_inventory(project_root=Path.cwd())

    outputs = write_dataset_inventory(rows=rows, docs_root=tmp_path, project_root=Path.cwd())

    text = (tmp_path / "dataset_inventory.md").read_text(encoding="utf-8")
    assert "C:\\" not in text
    assert "TODO: paste Google Drive URL after upload" in text
    assert "Heldout `*/test` MP4s remain validation-only" in text
    assert Path(outputs["json"]).exists()


def test_copy_verified_report_figures_copies_tcn_and_system_diagram(tmp_path: Path) -> None:
    tcn = tmp_path / "tcn.png"
    Image.new("RGB", (120, 80), (240, 240, 240)).save(tcn)

    outputs = copy_verified_report_figures(project_root=Path.cwd(), figure_root=tmp_path / "figures", tcn_image=tcn)

    assert Path(outputs["final_live_candidate_poster"]).exists()
    assert Path(outputs["canonical_skeleton_seed"]).exists()
    assert Path(outputs["real_guided_skeleton_augmentation"]).exists()
    assert Path(outputs["real_guided_alignment_error"]).exists()
    assert Path(outputs["schunk_paper_yaw45_pitch20"]).exists()
    assert Path(outputs["tcn_temporal_model"]).exists()
    assert Path(outputs["final_take_timing_budget"]).exists()
    assert Path(outputs["verified_system_pipeline"]).exists()
    with Image.open(Path(outputs["verified_system_pipeline"])) as pipeline:
        assert pipeline.width >= 2000
        assert pipeline.height >= 1400


def test_submission_manifest_keeps_v4_live_and_v7e_diagnostic(tmp_path: Path) -> None:
    rows = resolve_dataset_inventory(project_root=Path.cwd())
    notebook = write_final_project_notebook(project_root=tmp_path)
    manifest = write_submission_manifest(
        project_root=Path.cwd(),
        docs_root=tmp_path,
        figure_outputs={"figure": "conference-latex-template/IEEE-conference-template-062824/figures/example.png"},
        dataset_rows=rows,
        notebook_path=notebook,
    )

    payload = json.loads((tmp_path / "submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["path"].endswith("submission_manifest.json")
    assert "real-guided skeleton" in payload["research_focus"]
    assert payload["pipeline_evidence"]["real_clips"] == 20
    assert payload["pipeline_evidence"]["large_sharded_samples"] == 10000
    assert payload["runtime_environment"]["cpu"] == "Intel(R) Core(TM) i5-14400F"
    assert payload["runtime_environment"]["gpu"] == "NVIDIA GeForce RTX 4060"
    assert payload["final_take_timing"]["decision_frame"] == 62
    assert payload["final_take_timing"]["scope"].startswith("local end-to-end")
    assert payload["live_demo_policy"]["family"] == "v4"
    assert payload["live_demo_policy"]["v7e_policy"] == "diagnostics only; not promoted"
    assert "realtime_demo_rehearsal_20260616" not in payload["final_live_candidate"]["video"]


def test_scan_generated_text_rejects_local_absolute_paths(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("C:\\Users\\user\\Desktop\\secret", encoding="utf-8")

    with pytest.raises(ValueError, match="Forbidden"):
        scan_generated_text_for_forbidden_content(project_root=Path.cwd(), roots=[bad])


def test_validate_no_heldout_test_mp4_rejects_training_or_demo_usage() -> None:
    validate_no_heldout_test_mp4(Path("dataset/test/example.mp4"), usage="validation")

    with pytest.raises(ValueError, match="validation-only"):
        validate_no_heldout_test_mp4(Path("dataset/test/example.mp4"), usage="training")
    with pytest.raises(ValueError, match="validation-only"):
        validate_no_heldout_test_mp4(Path("dataset/test/example.mp4"), usage="demo")
