# Actuator-Constrained Early Intention Prediction for Simulated RPS Robot Hands

This project studies a skeleton-first real-to-sim-to-real pipeline for Rock-Paper-Scissors early intention prediction. A small number of real hand-motion videos are converted into MediaPipe 21-landmark skeleton trajectories, canonicalized, expanded through real-guided synthetic skeleton augmentation, aligned to SCHUNK/Isaac-style robot-hand pose progress, and transferred back to live camera input.

The simulated robot-hand counterattack is the final application and validation target. The core research contribution is the data and timing pipeline: scarce real skeleton seeds are expanded into a larger synthetic skeleton training set, then a temporal predictor is tested under a prompt-window actuator deadline.

## Research Pipeline

- Real seed capture: `20` reviewed RPS transition clips and `720` processed frames with MediaPipe hand detection coverage `1.0`.
- Canonical representation: `20` skeleton sequences with `142` temporal feature dimensions and max sequence length `56`.
- Real-guided augmentation: `2,000` compact samples and `10,000` large sharded samples derived from real skeleton seeds.
- Render-ready alignment: 32-frame SCHUNK/Isaac-style pose-progress alignment, reducing max progress error from about `0.106` to about `0.016`.
- Sim-to-real validation: frozen v4 temporal predictor applied back to live prompt-window capture, then checked by actuator feasibility before the robot counterattack is accepted.

## Final Sim-to-Real Demo Status

- Live/demo predictor: frozen v4 fallback policy.
- Final live candidate: `artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/selected_final_submission_take/final_submission_live_counterattack.mp4`
- Final selected take: `human rock -> robot paper`
- Result: `actuator_feasible_win`
- Supporting three-take review: `artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619/combined_three_take_review/final_submission_three_take_counterattack_compilation.mp4`
- Demo video upload: TODO: paste YouTube URL after upload.
- Presentation video upload: TODO: paste YouTube URL after upload.

The final demo timing is local desktop validation timing, including camera capture, MediaPipe, PyTorch/CUDA when available, OpenCV output, and actuator-feasibility checks. It is not a hardware-independent model latency benchmark.

The final candidate follows the required prompt timing:

```text
PROMPT ROCK -> PROMPT PAPER -> PROMPT SCISSORS
```

The human performs the target gesture during the final `PROMPT SCISSORS` response window.

## Repository Contents

- `src/embodied_rps/`: core dataset generation, skeleton prediction, policy, actuator feasibility, SCHUNK-style rendering, and final-submission helpers.
- `configs/`: frozen v4 live/demo config and dataset/model configs used during the project.
- `tests/`: focused regression tests for policies, feasibility, live recording, and submission artifacts.
- `docs/dataset_inventory.md`: local dataset roots for Google Drive upload.
- `docs/submission_manifest.json`: machine-readable final artifact manifest.
- `final-project.ipynb`: lightweight notebook that summarizes the verified final artifacts and validation commands.
- `conference-latex-template/IEEE-conference-template-062824/final_report.tex`: IEEE-style final report source.

Large generated datasets, videos, and intermediate experiment artifacts are intentionally excluded from GitHub. Upload the `upload_ready` dataset rows in `docs/dataset_inventory.md` to Google Drive and paste links below.

Dataset Google Drive link: TODO: paste Google Drive URL after upload.

## Setup

Use Python 3.10 or newer. From the repository root:

```powershell
python -m pip install -e .[dev]
```

For the live camera demo path, the local environment also needs a working camera and the OpenCV/MediaPipe stack installed by the package dependencies.

## Reproduce The Final Submission Artifacts

The final live candidate was already recorded. To regenerate documentation/report support files from the verified local artifacts, including the real-guided skeleton figures and final report manifest:

```powershell
$env:PYTHONPATH='src'
python -m embodied_rps.tools.build_final_submission_assets --tcn-image <path-to-tcn-diagram.png>
```

To inspect the frozen v4 live/demo policy:

```powershell
$env:PYTHONPATH='src'
python -m embodied_rps.tools.run_current_best_realtime_demo --dry-run --config configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml --output-root artifacts/final_submission_model_policy_freeze_20260619/dry_run
```

To run the final live recording workflow again, use the prompt sequence exactly as configured:

```powershell
$env:PYTHONPATH='src'
python -m embodied_rps.tools.run_final_submission_live_counterattack_recording --config configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml --pose-config configs/kinematic_rps.yaml --style-asset-root artifacts/schunk_joint_target_skeleton_passed --output-root artifacts/final_submission_live_counterattack_recording_prompt_scissors_retake_20260619 --camera 0 --max-frames 300
```

## Validation

Run the focused final-submission checks:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_current_best_realtime_demo_cli.py tests/test_final_robot_counterattack_demo.py tests/test_final_submission_live_counterattack_recording.py tests/test_policy.py tests/test_feasibility.py tests/test_final_submission_assets.py -q
python -m compileall src tests
```

Report build:

```powershell
cd conference-latex-template\IEEE-conference-template-062824
latexmk -pdf final_report.tex
```

## Model Policy

The live/demo model is v4 and remains frozen for the final demo:

```text
config = configs/realtime_two_stage_selector_demo_conditional_scissors_rescue_rock_guard.yaml
profiles = v4_fewshot_aug_tcn, v4_rebalanced_tcn, v4_final_gate_micro
weights = 0.25, 0.75, 0.0
confidence threshold = 0.70
margin threshold = 0.10
confirmation count = 2
binary transition threshold = 0.60
response prompt = scissors
```

v7e is preserved as a diagnostic branch only. It reached original20 strict validation `17/20` and is not promoted to the live/demo predictor. v7f retraining was not started.

## Dataset Handoff

See `docs/dataset_inventory.md`.

Upload-ready local roots include the core real-guided skeleton datasets and diagnostic follow-up datasets:

- `artifacts/real_guided_training_package_20260610/`
- `artifacts/real_guided_large_sharded_20260610/`
- `artifacts/real_guided_three_class_wait_expanded_v4_final_gate_micro_g1000_20260616/`
- `artifacts/real_guided_three_class_wait_expanded_v7e_stage1_paper_transition_rescue_20260619/` as diagnostics only
- `artifacts/real_guided_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_20260619/` as diagnostics only

The current `v4_fewshot_aug` and `v4_rebalanced` dataset roots in this workspace are metadata-only because the profile metadata exists but the shard files are not present here.

Heldout `*/test` MP4s are validation-only. They are not training inputs and are not demo-generation inputs.

## Report And Submission Links

- Final report source: `conference-latex-template/IEEE-conference-template-062824/final_report.tex`
- Final report PDF: `conference-latex-template/IEEE-conference-template-062824/final_report.pdf`
- Presentation slides: `presentation-slides.pdf`
- Proposal: `proposal.pdf`
- Demo video upload: TODO: paste YouTube URL after upload.
- Presentation video upload: TODO: paste YouTube URL after upload.
- Dataset upload: TODO: paste Google Drive URL after upload.

The protected files `proposal.pdf` and `presentation-slides.pdf` are preserved as existing source/submission artifacts and are not edited by this final repository preparation pass.
