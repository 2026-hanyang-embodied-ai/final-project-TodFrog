# Dataset Inventory For Google Drive Upload

All paths are workspace-relative. Large datasets should be uploaded to Google Drive and linked in the README.
Heldout `*/test` MP4s remain validation-only and are not training or demo-generation inputs.

| Name | Status | Role | Local path | Size | Google Drive URL |
| --- | --- | --- | --- | ---: | --- |
| compact_real_guided_training_package | `upload_ready` | `primary` | `artifacts/real_guided_training_package_20260610` | 52.81 MB | TODO: paste Google Drive URL after upload |
| large_real_guided_sharded_dataset | `upload_ready` | `supporting` | `artifacts/real_guided_large_sharded_20260610` | 274.12 MB | TODO: paste Google Drive URL after upload |
| v4_final_gate_micro_dataset | `upload_ready` | `supporting` | `artifacts/real_guided_three_class_wait_expanded_v4_final_gate_micro_g1000_20260616` | 401.76 MB | TODO: paste Google Drive URL after upload |
| v7e_three_class_diagnostic_dataset | `upload_ready` | `diagnostic` | `artifacts/real_guided_three_class_wait_expanded_v7e_stage1_paper_transition_rescue_20260619` | 1.54 GB | TODO: paste Google Drive URL after upload |
| v7e_stage1_rock_transition_diagnostic_dataset | `upload_ready` | `diagnostic` | `artifacts/real_guided_two_stage_rock_transition_v7e_stage1_paper_transition_rescue_20260619` | 1.45 GB | TODO: paste Google Drive URL after upload |
| v4_fewshot_aug_dataset_root | `metadata_only` | `metadata_only` | `artifacts/real_guided_three_class_wait_expanded_v4_fewshot_aug_20260615` | 3.47 KB | TODO: paste Google Drive URL after upload |
| v4_rebalanced_dataset_root | `metadata_only` | `metadata_only` | `artifacts/real_guided_three_class_wait_expanded_v4_rebalanced_20260615` | 4.05 KB | TODO: paste Google Drive URL after upload |

## Notes

- `upload_ready` means shard files are present in this workspace root.
- `metadata_only` means summaries/profile references exist, but shard files are not present here.
- v7e datasets are diagnostics only; they are not promoted to the live/demo predictor.
