"""Three-class skeleton dataset generation for paper-wait realtime policy."""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from embodied_rps.hard_example_skeletons import (
    apply_3d_view_transform,
    canonicalize_mediapipe_style,
    load_base_dataset_samples,
    split_for_index,
)
from embodied_rps.pose_family import (
    FINGER_NAMES,
    FingerName,
    Handedness,
    PersonIdentity,
    build_keypoints_from_semantics,
)
from embodied_rps.real_skeleton_training import landmark_velocity_features

TargetName: TypeAlias = Literal["rock", "paper", "scissors"]
SplitName: TypeAlias = Literal["train", "val", "test"]
AugmentationProfile: TypeAlias = Literal[
    "baseline",
    "v2_targeted",
    "v3_targeted",
    "v4_fewshot",
    "v4_contrastive",
    "v4_rebalanced",
    "v4_failure_focused",
    "v4_remaining_gate",
    "v4_selector_targets",
    "v4_temporal_curl",
    "v4_boundary_pairs",
    "v4_hard_paper_scissors",
    "v4_delayed_paper_timing",
    "v4_mixed_paper_timing",
    "v4_live_prompt_hard",
    "v4_final_gate_micro",
    "v4_paper_rescue_micro",
    "v4_prompt_wait_hard",
    "v7_rps_pose",
    "v7b_rps_pose_conservative_scissors",
    "v7c_prompt_window_rock_guarded_paper_rescue",
    "v7d_real_seeded_prompt_window_guard",
    "v7e_stage1_paper_transition_rescue",
]

TARGET_NAMES: tuple[TargetName, ...] = ("rock", "paper", "scissors")
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
TARGET_TO_TRANSITION: dict[TargetName, str] = {
    "rock": "rock_hold",
    "paper": "rock_to_paper",
    "scissors": "rock_to_scissors",
}
TARGET_TO_LABEL: dict[TargetName, int] = {"rock": 0, "paper": 1, "scissors": 2}
CORE_FIELDS: tuple[str, ...] = (
    "sample_ids",
    "labels",
    "label_names",
    "target_names",
    "split_names",
    "lengths",
    "mask",
    "progress",
    "canonical_landmarks",
    "features",
    "source_names",
    "hard_example_flags",
)
V7C_BEHAVIOR_PROFILE: AugmentationProfile = "v7c_prompt_window_rock_guarded_paper_rescue"
V7D_PROFILE: AugmentationProfile = "v7d_real_seeded_prompt_window_guard"
V7E_PROFILE: AugmentationProfile = "v7e_stage1_paper_transition_rescue"
PROMPT_CONDITIONED_PROFILES: frozenset[AugmentationProfile] = frozenset(
    {
        "v7b_rps_pose_conservative_scissors",
        V7C_BEHAVIOR_PROFILE,
        V7D_PROFILE,
        V7E_PROFILE,
    }
)
ROCK_GUARD_PAPER_RESCUE_PROFILES: frozenset[AugmentationProfile] = frozenset(
    {V7C_BEHAVIOR_PROFILE, V7D_PROFILE, V7E_PROFILE}
)
REAL_SEEDED_PROMPT_WINDOW_PROFILES: frozenset[AugmentationProfile] = frozenset({V7D_PROFILE, V7E_PROFILE})


@dataclass(frozen=True)
class ThreeClassWaitExpansionConfig:
    """Configuration for the 3-class paper-wait skeleton expansion."""

    output_root: Path
    base_dataset_root: Path
    generated_per_target: int = 2500
    sequence_length: int = 72
    min_length: int = 48
    shard_size: int = 512
    seed: int = 20260611
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    base_rock_stride: int = 2
    augmentation_profile: AugmentationProfile = "baseline"
    calibration_seed_package_root: Path | None = None
    live_rock_seed_package_root: Path | None = None
    v7_seed_package_root: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class ThreeClassSample:
    """One padded 3-class skeleton sample ready for sharding."""

    sample_id: str
    split: SplitName
    target_name: TargetName
    canonical_landmarks: NDArray[np.float32]
    mask: NDArray[np.bool_]
    progress: NDArray[np.float32]
    source_name: str
    hard_example: bool
    metadata: dict[str, object]

    @property
    def length(self) -> int:
        return int(np.sum(self.mask))

    @property
    def transition_label(self) -> str:
        return TARGET_TO_TRANSITION[self.target_name]

    @property
    def label_id(self) -> int:
        return TARGET_TO_LABEL[self.target_name]


def generate_three_class_wait_dataset(config: ThreeClassWaitExpansionConfig) -> dict[str, object]:
    """Generate, shard, validate, and document the 3-class wait-policy dataset."""

    _validate_config(config)
    _prepare_output(config.output_root, overwrite=config.overwrite)
    rng = np.random.default_rng(config.seed)
    samples: list[ThreeClassSample] = []
    base_samples = load_base_dataset_samples(config.base_dataset_root, sequence_length=config.sequence_length)
    samples.extend(_copy_binary_base_samples(base_samples))
    samples.extend(_rock_hold_samples_from_base(base_samples, stride=config.base_rock_stride))
    samples.extend(_copy_v4_calibration_seed_samples(config))
    live_rock_seed_samples = _copy_live_rock_false_trigger_seed_samples(config)
    samples.extend(live_rock_seed_samples)
    samples.extend(_live_rock_false_trigger_balanced_controls(live_rock_seed_samples, config=config, rng=rng))
    v7_seed_samples = _copy_v7_rps_seed_samples(config)
    samples.extend(v7_seed_samples)
    samples.extend(_v7_real_seed_balanced_controls(v7_seed_samples, config=config, rng=rng))
    samples.extend(generate_three_class_hard_samples(config, rng=rng))
    samples = sorted(samples, key=lambda sample: (sample.split, sample.target_name, sample.sample_id))
    shard_rows = write_three_class_shards(
        config.output_root,
        samples,
        shard_size=config.shard_size,
        sequence_length=config.sequence_length,
    )
    metadata_path = write_three_class_sample_metadata(config.output_root, samples)
    validation = validate_three_class_wait_dataset(
        config.output_root,
        shard_rows=shard_rows,
        expected_sample_count=len(samples),
    )
    write_csv(config.output_root / "shard_index.csv", shard_rows, fieldnames=list(shard_rows[0].keys()) if shard_rows else [])
    write_json(config.output_root / "validation_summary.json", validation)
    write_json(config.output_root / "generation_config.json", _json_ready(asdict(config)))
    dataset_card = write_three_class_dataset_card(config.output_root, validation)
    run_summary = {
        "status": validation["status"],
        "output_root": config.output_root.as_posix(),
        "sample_count": len(samples),
        "generated_per_target": config.generated_per_target,
        "base_dataset_root": config.base_dataset_root.as_posix(),
        "dataset_card": dataset_card.as_posix(),
        "sample_metadata": metadata_path.as_posix(),
        "validation_summary": (config.output_root / "validation_summary.json").as_posix(),
        "validation": validation,
    }
    write_json(config.output_root / "run_summary.json", run_summary)
    return run_summary


def generate_three_class_hard_samples(
    config: ThreeClassWaitExpansionConfig,
    *,
    rng: np.random.Generator,
) -> list[ThreeClassSample]:
    """Generate balanced procedural rock, paper, and scissors samples."""

    samples: list[ThreeClassSample] = []
    for target_name in TARGET_NAMES:
        for index in range(config.generated_per_target):
            split = split_for_index(index, config.generated_per_target, train_fraction=config.train_fraction, val_fraction=config.val_fraction)
            samples.append(
                generate_one_three_class_sample(
                    target_name=target_name,
                    sample_index=index,
                    split=split,
                    rng=rng,
                    sequence_length=config.sequence_length,
                    min_length=config.min_length,
                    augmentation_profile=config.augmentation_profile,
                )
            )
    return samples


def generate_one_three_class_sample(
    *,
    target_name: TargetName,
    sample_index: int,
    split: SplitName,
    rng: np.random.Generator,
    sequence_length: int,
    min_length: int,
    augmentation_profile: AugmentationProfile = "baseline",
) -> ThreeClassSample:
    """Generate one procedural 3D canonical skeleton trajectory."""

    behavior_profile = _behavior_profile(augmentation_profile)
    length = _sample_length(target_name, rng, sequence_length=sequence_length, min_length=min_length, augmentation_profile=behavior_profile)
    mask = np.zeros((sequence_length,), dtype=np.bool_)
    mask[:length] = True
    progress = np.zeros((sequence_length,), dtype=np.float32)
    progress[:length] = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if length < sequence_length:
        progress[length:] = 1.0

    identity = _sample_identity(rng, person_id=sample_index)
    yaw, pitch, roll = _sample_viewpoint(target_name, rng, augmentation_profile=behavior_profile)
    global_scale = (
        float(rng.uniform(0.68, 1.42))
        if behavior_profile in {
            "v4_live_prompt_hard",
            "v4_final_gate_micro",
            "v4_paper_rescue_micro",
            "v4_prompt_wait_hard",
            "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue",
        }
        else
        float(rng.uniform(0.68, 1.42))
        if augmentation_profile == "v4_selector_targets"
        else
        float(rng.uniform(0.70, 1.40))
        if augmentation_profile == "v4_hard_paper_scissors"
        else
        float(rng.uniform(0.70, 1.40))
        if augmentation_profile == "v4_boundary_pairs"
        else
        float(rng.uniform(0.70, 1.40))
        if augmentation_profile == "v4_remaining_gate"
        else
        float(rng.uniform(0.70, 1.38))
        if augmentation_profile == "v4_failure_focused"
        else float(rng.uniform(0.72, 1.34))
        if augmentation_profile == "v4_rebalanced"
        else float(rng.uniform(0.70, 1.40))
        if augmentation_profile == "v4_contrastive"
        else float(rng.uniform(0.72, 1.36))
        if augmentation_profile == "v4_fewshot"
        else float(rng.uniform(0.76, 1.30))
        if augmentation_profile == "v2_targeted"
        else float(rng.uniform(0.80, 1.24))
    )
    depth_scale = (
        float(rng.uniform(0.62, 1.42))
        if behavior_profile in {
            "v4_live_prompt_hard",
            "v4_final_gate_micro",
            "v4_paper_rescue_micro",
            "v4_prompt_wait_hard",
            "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue",
        }
        else
        float(rng.uniform(0.62, 1.42))
        if augmentation_profile == "v4_selector_targets"
        else
        float(rng.uniform(0.64, 1.40))
        if augmentation_profile == "v4_hard_paper_scissors"
        else
        float(rng.uniform(0.64, 1.40))
        if augmentation_profile == "v4_boundary_pairs"
        else
        float(rng.uniform(0.64, 1.40))
        if augmentation_profile == "v4_remaining_gate"
        else
        float(rng.uniform(0.66, 1.38))
        if augmentation_profile == "v4_failure_focused"
        else float(rng.uniform(0.70, 1.34))
        if augmentation_profile == "v4_rebalanced"
        else float(rng.uniform(0.66, 1.40))
        if augmentation_profile == "v4_contrastive"
        else float(rng.uniform(0.68, 1.36))
        if augmentation_profile == "v4_fewshot"
        else float(rng.uniform(0.76, 1.28))
        if augmentation_profile == "v2_targeted"
        else float(rng.uniform(0.84, 1.20))
    )
    translation = cast(NDArray[np.float32], rng.normal(0.0, 0.035, size=(3,)).astype(np.float32))
    if behavior_profile in {
        "v4_contrastive",
        "v4_rebalanced",
        "v4_failure_focused",
        "v4_remaining_gate",
        "v4_selector_targets",
        "v4_temporal_curl",
        "v4_boundary_pairs",
        "v4_hard_paper_scissors",
        "v4_delayed_paper_timing",
        "v4_mixed_paper_timing",
        "v4_live_prompt_hard",
        "v4_final_gate_micro",
        "v4_paper_rescue_micro",
        "v4_prompt_wait_hard",
        "v7_rps_pose",
        "v7b_rps_pose_conservative_scissors",
        "v7c_prompt_window_rock_guarded_paper_rescue",
        "v7d_real_seeded_prompt_window_guard",
    }:
        pair_seed_base = (
            2026061820
            if augmentation_profile == V7D_PROFILE
            else
            2026061810
            if behavior_profile == V7C_BEHAVIOR_PROFILE
            else
            2026061800
            if behavior_profile == "v7b_rps_pose_conservative_scissors"
            else
            2026061700
            if behavior_profile == "v7_rps_pose"
            else
            2026061630
            if behavior_profile == "v4_prompt_wait_hard"
            else
            2026061620
            if behavior_profile == "v4_paper_rescue_micro"
            else
            2026061610
            if behavior_profile == "v4_final_gate_micro"
            else
            2026061600
            if behavior_profile == "v4_live_prompt_hard"
            else
            2026061590
            if behavior_profile == "v4_mixed_paper_timing"
            else
            2026061580
            if behavior_profile == "v4_delayed_paper_timing"
            else
            2026061550
            if behavior_profile == "v4_temporal_curl"
            else
            2026061570
            if behavior_profile == "v4_hard_paper_scissors"
            else
            2026061560
            if behavior_profile == "v4_boundary_pairs"
            else
            2026061540
            if behavior_profile == "v4_selector_targets"
            else
            2026061530
            if behavior_profile == "v4_remaining_gate"
            else 2026061520
            if behavior_profile == "v4_failure_focused"
            else 2026061500
            if behavior_profile == "v4_contrastive"
            else 2026061510
        )
        pair_rng = np.random.default_rng(pair_seed_base + sample_index)
        identity = _sample_identity(pair_rng, person_id=sample_index)
        yaw, pitch, roll = _sample_viewpoint(target_name, pair_rng, augmentation_profile=behavior_profile)
        if behavior_profile in {
            "v4_live_prompt_hard",
            "v4_final_gate_micro",
            "v4_paper_rescue_micro",
            "v4_prompt_wait_hard",
            "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue",
        }:
            global_scale = float(pair_rng.uniform(0.68, 1.42))
            depth_scale = float(pair_rng.uniform(0.62, 1.42))
        elif behavior_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"}:
            global_scale = float(pair_rng.uniform(0.70, 1.40))
            depth_scale = float(pair_rng.uniform(0.64, 1.40))
        elif behavior_profile == "v4_temporal_curl":
            global_scale = float(pair_rng.uniform(0.70, 1.38))
            depth_scale = float(pair_rng.uniform(0.64, 1.40))
        elif behavior_profile == "v4_hard_paper_scissors":
            global_scale = float(pair_rng.uniform(0.70, 1.40))
            depth_scale = float(pair_rng.uniform(0.64, 1.40))
        elif behavior_profile == "v4_boundary_pairs":
            global_scale = float(pair_rng.uniform(0.70, 1.40))
            depth_scale = float(pair_rng.uniform(0.64, 1.40))
        elif behavior_profile == "v4_selector_targets":
            global_scale = float(pair_rng.uniform(0.68, 1.42))
            depth_scale = float(pair_rng.uniform(0.62, 1.42))
        elif behavior_profile == "v4_remaining_gate":
            global_scale = float(pair_rng.uniform(0.70, 1.40))
            depth_scale = float(pair_rng.uniform(0.64, 1.40))
        elif behavior_profile == "v4_failure_focused":
            global_scale = float(pair_rng.uniform(0.70, 1.38))
            depth_scale = float(pair_rng.uniform(0.66, 1.38))
        elif behavior_profile == "v4_rebalanced":
            global_scale = float(pair_rng.uniform(0.72, 1.34))
            depth_scale = float(pair_rng.uniform(0.70, 1.34))
        else:
            global_scale = float(pair_rng.uniform(0.70, 1.40))
            depth_scale = float(pair_rng.uniform(0.66, 1.40))
        translation = cast(NDArray[np.float32], pair_rng.normal(0.0, 0.035, size=(3,)).astype(np.float32))
    noise_std = _sample_observation_noise(target_name, rng, augmentation_profile=behavior_profile)
    start_curls, start_spreads = _start_pose(target_name, rng, augmentation_profile=behavior_profile)
    final_curls, final_spreads = _target_pose(target_name, rng, augmentation_profile=behavior_profile)
    onset = _finger_onsets(target_name, rng, augmentation_profile=behavior_profile, sample_index=sample_index)
    hesitation_center, hesitation_width, hesitation_strength = _hesitation_params(
        target_name,
        rng,
        augmentation_profile=behavior_profile,
    )
    wobble = _wobble_params(target_name, rng, augmentation_profile=behavior_profile)

    canonical = np.zeros((sequence_length, 21, 3), dtype=np.float32)
    last_valid = np.zeros((21, 3), dtype=np.float32)
    for frame_index in range(sequence_length):
        frame_progress = float(progress[min(frame_index, length - 1)])
        curls: dict[FingerName, float] = {}
        spreads: dict[FingerName, float] = {}
        for finger in FINGER_NAMES:
            motion = _motion_progress(
                frame_progress,
                onset=onset[finger],
                hesitation_center=hesitation_center,
                hesitation_width=hesitation_width,
                hesitation_strength=hesitation_strength,
            )
            curls[finger] = start_curls[finger] + (final_curls[finger] - start_curls[finger]) * motion
            spreads[finger] = start_spreads[finger] + (final_spreads[finger] - start_spreads[finger]) * motion
        raw = build_keypoints_from_semantics(curls, spreads, identity)
        if identity.handedness == "left":
            raw = raw.copy()
            raw[:, 0] *= -1.0
        dynamic_yaw, dynamic_pitch, dynamic_roll, dynamic_translation = _dynamic_view(
            yaw,
            pitch,
            roll,
            translation,
            frame_index=frame_index,
            frame_progress=frame_progress,
            rng=rng,
            wobble=wobble,
        )
        transformed = apply_3d_view_transform(
            raw * np.float32(global_scale),
            yaw_degrees=dynamic_yaw,
            pitch_degrees=dynamic_pitch,
            roll_degrees=dynamic_roll,
            depth_scale=depth_scale,
        )
        transformed = transformed + dynamic_translation
        if frame_index < length:
            transformed = transformed + cast(NDArray[np.float32], rng.normal(0.0, noise_std, size=transformed.shape).astype(np.float32))
            last_valid = _apply_canonical_observation_noise(
                canonicalize_mediapipe_style(transformed),
                target_name=target_name,
                frame_progress=frame_progress,
                rng=rng,
                augmentation_profile=behavior_profile,
            )
            canonical[frame_index] = last_valid
        else:
            canonical[frame_index] = last_valid

    sample_prefix = (
        "v7e_stage1_paper_transition_rescue_three_class"
        if augmentation_profile == V7E_PROFILE
        else
        "v7d_real_seeded_prompt_window_guard_three_class"
        if augmentation_profile == V7D_PROFILE
        else
        "v7c_prompt_window_rock_guarded_paper_rescue_three_class"
        if augmentation_profile == V7C_BEHAVIOR_PROFILE
        else
        "v7b_rps_pose_conservative_scissors_three_class"
        if augmentation_profile == "v7b_rps_pose_conservative_scissors"
        else
        "v7_rps_pose_three_class"
        if augmentation_profile == "v7_rps_pose"
        else
        "v4_prompt_wait_hard_three_class"
        if augmentation_profile == "v4_prompt_wait_hard"
        else
        "v4_paper_rescue_micro_three_class"
        if augmentation_profile == "v4_paper_rescue_micro"
        else
        "v4_final_gate_micro_three_class"
        if augmentation_profile == "v4_final_gate_micro"
        else
        "v4_live_prompt_hard_three_class"
        if augmentation_profile == "v4_live_prompt_hard"
        else
        "v4_mixed_paper_timing_three_class"
        if augmentation_profile == "v4_mixed_paper_timing"
        else
        "v4_delayed_paper_timing_three_class"
        if augmentation_profile == "v4_delayed_paper_timing"
        else
        "v4_hard_paper_scissors_three_class"
        if augmentation_profile == "v4_hard_paper_scissors"
        else
        "v4_boundary_pairs_three_class"
        if augmentation_profile == "v4_boundary_pairs"
        else
        "v4_temporal_curl_three_class"
        if augmentation_profile == "v4_temporal_curl"
        else
        "v4_selector_targets_three_class"
        if augmentation_profile == "v4_selector_targets"
        else
        "v4_remaining_gate_three_class"
        if augmentation_profile == "v4_remaining_gate"
        else
        "v4_failure_focused_three_class"
        if augmentation_profile == "v4_failure_focused"
        else "v4_rebalanced_three_class"
        if augmentation_profile == "v4_rebalanced"
        else "v4_contrastive_three_class"
        if augmentation_profile == "v4_contrastive"
        else "v4_fewshot_three_class"
        if augmentation_profile == "v4_fewshot"
        else "v3_three_class"
        if augmentation_profile == "v3_targeted"
        else "v2_three_class"
        if augmentation_profile == "v2_targeted"
        else "three_class"
    )
    sample_id = f"{sample_prefix}_{TARGET_TO_TRANSITION[target_name]}_{sample_index + 1:05d}"
    return ThreeClassSample(
        sample_id=sample_id,
        split=split,
        target_name=target_name,
        canonical_landmarks=canonical,
        mask=mask,
        progress=progress,
        source_name=_source_name(target_name, augmentation_profile, sample_index=sample_index),
        hard_example=True,
        metadata={
            "target_name": target_name,
            "transition_label": TARGET_TO_TRANSITION[target_name],
            "generated_length": length,
            "handedness": identity.handedness,
            "augmentation_profile": augmentation_profile,
            "yaw_degrees": yaw,
            "pitch_degrees": pitch,
            "roll_degrees": roll,
            "global_scale": global_scale,
            "depth_scale": depth_scale,
            "noise_std": noise_std,
            "wobble": wobble,
            "start_curls": start_curls,
            "start_spreads": start_spreads,
            "final_curls": final_curls,
            "final_spreads": final_spreads,
            "onsets": onset,
            "paper_timing_mode": _paper_timing_mode(
                target_name,
                augmentation_profile=augmentation_profile,
                sample_index=sample_index,
            ),
            "hard_case": _hard_case_name(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index),
            "contrastive_boundary": augmentation_profile
            in {
                "v4_contrastive",
                "v4_rebalanced",
                "v4_failure_focused",
                "v4_remaining_gate",
                "v4_selector_targets",
                "v4_temporal_curl",
                "v4_boundary_pairs",
                "v4_hard_paper_scissors",
                "v4_delayed_paper_timing",
                "v4_mixed_paper_timing",
                "v4_live_prompt_hard",
                "v4_final_gate_micro",
                "v4_paper_rescue_micro",
                "v4_prompt_wait_hard",
                "v7_rps_pose",
                "v7b_rps_pose_conservative_scissors",
                "v7c_prompt_window_rock_guarded_paper_rescue",
                "v7d_real_seeded_prompt_window_guard",
                "v7e_stage1_paper_transition_rescue",
            },
            "paper_recovery_profile": augmentation_profile
            in {
                "v4_rebalanced",
                "v4_failure_focused",
                "v4_remaining_gate",
                "v4_selector_targets",
                "v4_temporal_curl",
                "v4_boundary_pairs",
                "v4_hard_paper_scissors",
                "v4_delayed_paper_timing",
                "v4_mixed_paper_timing",
                "v4_live_prompt_hard",
                "v4_final_gate_micro",
                "v4_paper_rescue_micro",
                "v4_prompt_wait_hard",
                "v7_rps_pose",
                "v7b_rps_pose_conservative_scissors",
                "v7c_prompt_window_rock_guarded_paper_rescue",
                "v7d_real_seeded_prompt_window_guard",
                "v7e_stage1_paper_transition_rescue",
            }
            and target_name == "paper",
            "late_scissors_recovery_profile": augmentation_profile
            in {
                "v4_failure_focused",
                "v4_remaining_gate",
                "v4_selector_targets",
                "v4_temporal_curl",
                "v4_boundary_pairs",
                "v4_hard_paper_scissors",
                "v4_delayed_paper_timing",
                "v4_mixed_paper_timing",
                "v4_live_prompt_hard",
                "v4_final_gate_micro",
                "v4_paper_rescue_micro",
                "v4_prompt_wait_hard",
                "v7_rps_pose",
                "v7b_rps_pose_conservative_scissors",
                "v7c_prompt_window_rock_guarded_paper_rescue",
                "v7d_real_seeded_prompt_window_guard",
                "v7e_stage1_paper_transition_rescue",
            }
            and target_name == "scissors",
            "remaining_gate_profile": augmentation_profile == "v4_remaining_gate",
            "selector_failure_target_profile": augmentation_profile == "v4_selector_targets",
            "temporal_curl_profile": augmentation_profile == "v4_temporal_curl",
            "boundary_pair_profile": augmentation_profile == "v4_boundary_pairs",
            "hard_paper_scissors_profile": augmentation_profile == "v4_hard_paper_scissors",
            "delayed_paper_timing_profile": augmentation_profile == "v4_delayed_paper_timing",
            "mixed_paper_timing_profile": augmentation_profile == "v4_mixed_paper_timing",
            "live_prompt_hard_profile": augmentation_profile == "v4_live_prompt_hard",
            "final_gate_micro_profile": augmentation_profile == "v4_final_gate_micro",
            "paper_rescue_micro_profile": augmentation_profile == "v4_paper_rescue_micro",
            "prompt_wait_hard_profile": augmentation_profile == "v4_prompt_wait_hard",
            "v7_rps_pose_profile": augmentation_profile == "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors_profile": augmentation_profile == "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue_profile": augmentation_profile
            == V7C_BEHAVIOR_PROFILE,
            "v7d_real_seeded_prompt_window_guard_profile": augmentation_profile == V7D_PROFILE,
            "v7e_stage1_paper_transition_rescue_profile": augmentation_profile == V7E_PROFILE,
            "real_seeded_prompt_window_guard_profile": augmentation_profile in REAL_SEEDED_PROMPT_WINDOW_PROFILES,
            "prompt_conditioned_sequence_profile": augmentation_profile in PROMPT_CONDITIONED_PROFILES,
            "rock_guard_profile": augmentation_profile in ROCK_GUARD_PAPER_RESCUE_PROFILES and target_name == "rock",
            "paper_rescue_profile": augmentation_profile in ROCK_GUARD_PAPER_RESCUE_PROFILES and target_name == "paper",
            "binary_transition_allowed": not (
                augmentation_profile in ROCK_GUARD_PAPER_RESCUE_PROFILES and target_name == "rock"
            ),
            "anti_scissors_negative_pressure": "early_closed_or_rock_like_frames_reject_scissors"
            if augmentation_profile in ROCK_GUARD_PAPER_RESCUE_PROFILES and target_name == "rock"
            else None,
            "static_early_scissors_positive": False
            if augmentation_profile in ROCK_GUARD_PAPER_RESCUE_PROFILES and target_name == "scissors"
            else None,
            "prompt_window_model": "standby_or_rock_like_prefix_then_bounded_response"
            if augmentation_profile in PROMPT_CONDITIONED_PROFILES
            else None,
            "heldout_policy": "held-out 15 MP4s remain validation-only",
            "held_out_reference_policy": "new 15 MP4s are diagnostic references only; held-out 15 MP4s remain validation-only and are not copied into training shards",
        },
    )


def write_three_class_shards(
    output_root: Path,
    samples: list[ThreeClassSample],
    *,
    shard_size: int,
    sequence_length: int,
) -> list[dict[str, object]]:
    """Write 3-class samples to split shards and return shard index rows."""

    shard_rows: list[dict[str, object]] = []
    for split in SPLIT_NAMES:
        split_samples = [sample for sample in samples if sample.split == split]
        for shard_index, start in enumerate(range(0, len(split_samples), shard_size)):
            shard_samples = split_samples[start : start + shard_size]
            shard_dir = output_root / "shards" / split
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"{split}_shard_{shard_index:03d}.npz"
            arrays = _arrays_for_samples(shard_samples, sequence_length=sequence_length)
            np.savez_compressed(shard_path, **arrays)
            row: dict[str, object] = {
                "split": split,
                "shard_index": shard_index,
                "path": shard_path.as_posix(),
                "sample_count": len(shard_samples),
                "valid_frame_count": int(np.sum(arrays["mask"])),
                "target_counts": dict(sorted(Counter(str(value) for value in arrays["target_names"].tolist()).items())),
                "hard_example_count": int(np.sum(arrays["hard_example_flags"])),
            }
            shard_rows.append(row)
    return shard_rows


def validate_three_class_wait_dataset(
    output_root: Path,
    *,
    shard_rows: list[dict[str, object]],
    expected_sample_count: int,
) -> dict[str, object]:
    """Validate the 3-class shard contract and canonical invariants."""

    failures: list[str] = []
    target_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    lengths: list[int] = []
    hard_count = 0
    wrist_errors: list[float] = []
    wrist_middle_scales: list[float] = []
    yaw_values: list[float] = []
    pitch_values: list[float] = []
    roll_values: list[float] = []
    sample_ids: list[str] = []

    for row in shard_rows:
        shard_path = Path(str(row["path"]))
        if not shard_path.exists():
            failures.append(f"missing shard {shard_path}")
            continue
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = set(CORE_FIELDS) - set(shard.files)
            if missing:
                failures.append(f"{shard_path} missing fields {sorted(missing)}")
                continue
            landmarks = cast(NDArray[np.float32], np.asarray(shard["canonical_landmarks"], dtype=np.float32))
            mask = cast(NDArray[np.bool_], np.asarray(shard["mask"], dtype=np.bool_))
            features = cast(NDArray[np.float32], np.asarray(shard["features"], dtype=np.float32))
            if landmarks.ndim != 4 or landmarks.shape[1:] != (72, 21, 3):
                failures.append(f"{shard_path} has unexpected canonical_landmarks shape {landmarks.shape}")
            if features.ndim != 3 or features.shape[1:] != (72, 126):
                failures.append(f"{shard_path} has unexpected features shape {features.shape}")
            if not np.all(np.isfinite(landmarks[mask])):
                failures.append(f"{shard_path} contains non-finite valid landmarks")
            if not np.all(np.isfinite(features[mask])):
                failures.append(f"{shard_path} contains non-finite valid features")
            sample_ids.extend(_decode_string_array(np.asarray(shard["sample_ids"])))
            targets = _decode_string_array(np.asarray(shard["target_names"]))
            splits = _decode_string_array(np.asarray(shard["split_names"]))
            sources = _decode_string_array(np.asarray(shard["source_names"]))
            target_counts.update(targets)
            split_counts.update(splits)
            source_counts.update(sources)
            lengths.extend(int(value) for value in np.asarray(shard["lengths"], dtype=np.int64).tolist())
            hard_flags = np.asarray(shard["hard_example_flags"], dtype=np.bool_)
            hard_count += int(np.sum(hard_flags))
            valid_landmarks = landmarks[mask]
            if valid_landmarks.size:
                wrist_errors.extend(np.linalg.norm(valid_landmarks[:, 0, :], axis=1).astype(float).tolist())
                wrist_middle_scales.extend(np.linalg.norm(valid_landmarks[:, 9, :] - valid_landmarks[:, 0, :], axis=1).astype(float).tolist())

    metadata_path = output_root / "sample_metadata.jsonl"
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if str(record.get("source_name", "")).startswith(("three_class", "v2_", "v3_", "v4_", "v7_", "v7d_")):
                yaw_values.append(float(record.get("yaw_degrees", 0.0)))
                pitch_values.append(float(record.get("pitch_degrees", 0.0)))
                roll_values.append(float(record.get("roll_degrees", 0.0)))
    else:
        failures.append("missing sample_metadata.jsonl")

    if len(sample_ids) != expected_sample_count:
        failures.append(f"expected {expected_sample_count} samples, got {len(sample_ids)}")
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("sample IDs are not unique")
    if set(target_counts) != set(TARGET_NAMES):
        failures.append(f"unexpected target labels {dict(target_counts)}")
    if len(set(target_counts.values())) > 1:
        failures.append(f"target counts are not balanced: {dict(target_counts)}")
    if wrist_errors and max(wrist_errors) > 1.0e-4:
        failures.append(f"wrist origin invariant failed: max {max(wrist_errors)}")

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "sample_count": len(sample_ids),
        "target_counts": dict(sorted(target_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "hard_example_count": hard_count,
        "hard_example_ratio": float(hard_count / max(1, len(sample_ids))),
        "length_min": int(min(lengths)) if lengths else None,
        "length_max": int(max(lengths)) if lengths else None,
        "length_mean": float(np.mean(lengths)) if lengths else None,
        "wrist_origin_max_error": float(max(wrist_errors)) if wrist_errors else None,
        "wrist_to_middle_mcp_scale_mean": float(np.mean(wrist_middle_scales)) if wrist_middle_scales else None,
        "viewpoint_distribution": {
            "yaw_min": float(min(yaw_values)) if yaw_values else None,
            "yaw_max": float(max(yaw_values)) if yaw_values else None,
            "pitch_min": float(min(pitch_values)) if pitch_values else None,
            "pitch_max": float(max(pitch_values)) if pitch_values else None,
            "roll_min": float(min(roll_values)) if roll_values else None,
            "roll_max": float(max(roll_values)) if roll_values else None,
        },
    }


def write_three_class_sample_metadata(output_root: Path, samples: list[ThreeClassSample]) -> Path:
    """Write one metadata JSONL record per 3-class sample."""

    path = output_root / "sample_metadata.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "target_name": sample.target_name,
                "transition_label": sample.transition_label,
                "length": sample.length,
                "source_name": sample.source_name,
                "hard_example": sample.hard_example,
                **sample.metadata,
            }
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_three_class_dataset_card(output_root: Path, validation: dict[str, object]) -> Path:
    """Write a concise dataset card for the 3-class wait-policy artifact."""

    path = output_root / "dataset_card.md"
    text = f"""# 3-Class Paper-Wait Skeleton Dataset, 2026-06-11

This dataset preserves the MediaPipe-compatible canonical skeleton contract and adds an explicit `rock` class for the paper-wait policy.

## Status

- validation: `{validation["status"]}`
- samples: `{validation["sample_count"]}`
- target counts: `{validation["target_counts"]}`
- split counts: `{validation["split_counts"]}`
- source counts: `{validation["source_counts"]}`
- hard example ratio: `{validation["hard_example_ratio"]}`

## Contract

Each shard contains padded `canonical_landmarks` with shape `(N, 72, 21, 3)`, `lengths`, `mask`, `progress`, `target_names`, and `sample_ids`.

Training features remain `landmark_velocity_126`: flattened `21 * 3` canonical landmarks plus frame-to-frame velocity.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def write_json(path: Path, data: object) -> None:
    """Write indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_binary_base_samples(base_samples: list[object]) -> list[ThreeClassSample]:
    copied: list[ThreeClassSample] = []
    for index, sample in enumerate(base_samples):
        target_name = _target_name(str(getattr(sample, "target_name")))
        if target_name == "rock":
            continue
        copied.append(
            ThreeClassSample(
                sample_id=f"base_{index:05d}_{target_name}",
                split=_split_name(str(getattr(sample, "split"))),
                target_name=target_name,
                canonical_landmarks=cast(NDArray[np.float32], getattr(sample, "canonical_landmarks")).astype(np.float32, copy=True),
                mask=cast(NDArray[np.bool_], getattr(sample, "mask")).astype(np.bool_, copy=True),
                progress=cast(NDArray[np.float32], getattr(sample, "progress")).astype(np.float32, copy=True),
                source_name="base_real_guided_binary",
                hard_example=False,
                metadata={"source_sample_id": str(getattr(sample, "sample_id"))},
            )
        )
    return copied


def _rock_hold_samples_from_base(base_samples: list[object], *, stride: int) -> list[ThreeClassSample]:
    rock_samples: list[ThreeClassSample] = []
    for index, sample in enumerate(base_samples):
        if index % stride != 0:
            continue
        landmarks = cast(NDArray[np.float32], getattr(sample, "canonical_landmarks")).astype(np.float32, copy=True)
        mask = cast(NDArray[np.bool_], getattr(sample, "mask")).astype(np.bool_, copy=True)
        progress = cast(NDArray[np.float32], getattr(sample, "progress")).astype(np.float32, copy=True)
        valid_length = int(np.sum(mask))
        cutoff = max(3, min(valid_length, int(math.ceil(valid_length * 0.22))))
        hold_frame = landmarks[cutoff - 1].copy()
        landmarks[cutoff:] = hold_frame
        rock_samples.append(
            ThreeClassSample(
                sample_id=f"rock_base_{index:05d}",
                split=_split_name(str(getattr(sample, "split"))),
                target_name="rock",
                canonical_landmarks=landmarks,
                mask=mask,
                progress=progress,
                source_name="base_early_fist_hold",
                hard_example=True,
                metadata={
                    "source_sample_id": str(getattr(sample, "sample_id")),
                    "source_target_name": str(getattr(sample, "target_name")),
                    "rock_hold_cutoff_frame": cutoff - 1,
                },
            )
        )
    return rock_samples


def _copy_v4_calibration_seed_samples(config: ThreeClassWaitExpansionConfig) -> list[ThreeClassSample]:
    if config.calibration_seed_package_root is None:
        return []
    seed_npz = config.calibration_seed_package_root / "v4_calibration_seed_dataset.npz"
    summary_path = config.calibration_seed_package_root / "seed_package_summary.json"
    if not seed_npz.exists():
        raise FileNotFoundError(f"Missing v4 calibration seed NPZ: {seed_npz}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise ValueError(f"v4 calibration seed package is not passed: {summary_path}")
    seed_samples: list[ThreeClassSample] = []
    with np.load(seed_npz, allow_pickle=False) as data:
        landmarks = cast(NDArray[np.float32], np.asarray(data["canonical_landmarks"], dtype=np.float32))
        mask = cast(NDArray[np.bool_], np.asarray(data["mask"], dtype=np.bool_))
        progress = cast(NDArray[np.float32], np.asarray(data["progress"], dtype=np.float32))
        lengths = cast(NDArray[np.int64], np.asarray(data["lengths"], dtype=np.int64))
        target_names = _decode_string_array(np.asarray(data["target_names"]))
        sample_ids = _decode_string_array(np.asarray(data["sample_ids"]))
        source_paths = _decode_string_array(np.asarray(data["source_paths"])) if "source_paths" in data else [""] * len(sample_ids)
        landmarks_json = _decode_string_array(np.asarray(data["landmarks_json"])) if "landmarks_json" in data else [""] * len(sample_ids)
    if landmarks.ndim != 4 or landmarks.shape[1:] != (config.sequence_length, 21, 3):
        raise ValueError(f"{seed_npz} canonical_landmarks must have shape (N,{config.sequence_length},21,3)")
    seed_count = int(landmarks.shape[0])
    if mask.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} mask must have shape (N,{config.sequence_length})")
    if progress.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} progress must have shape (N,{config.sequence_length})")
    if lengths.shape != (seed_count,):
        raise ValueError(f"{seed_npz} lengths must have shape (N,)")
    if len(target_names) != seed_count or len(sample_ids) != seed_count:
        raise ValueError(f"{seed_npz} target_names/sample_ids length mismatch")
    for index in range(seed_count):
        target_name = _target_name(target_names[index])
        split = split_for_index(index, seed_count, train_fraction=config.train_fraction, val_fraction=config.val_fraction)
        seed_samples.append(
            ThreeClassSample(
                sample_id=f"seed_{sample_ids[index]}",
                split=split,
                target_name=target_name,
                canonical_landmarks=landmarks[index].astype(np.float32, copy=True),
                mask=mask[index].astype(np.bool_, copy=True),
                progress=progress[index].astype(np.float32, copy=True),
                source_name="v4_calibration_real_seed",
                hard_example=False,
                metadata={
                    "source_seed_sample_id": sample_ids[index],
                    "source_path": source_paths[index] if index < len(source_paths) else "",
                    "landmarks_json": landmarks_json[index] if index < len(landmarks_json) else "",
                    "seed_package_root": config.calibration_seed_package_root.as_posix(),
                    "source_length": int(lengths[index]),
                    "augmentation_role": "real_seed_anchor",
                    "heldout_policy": "v4 calibration seeds are non-held-out only; the held-out test root remains excluded.",
                },
            )
        )
    return seed_samples


def _copy_live_rock_false_trigger_seed_samples(config: ThreeClassWaitExpansionConfig) -> list[ThreeClassSample]:
    if config.live_rock_seed_package_root is None:
        return []
    seed_npz = config.live_rock_seed_package_root / "live_rock_false_trigger_seed_dataset.npz"
    summary_path = config.live_rock_seed_package_root / "seed_package_summary.json"
    if not seed_npz.exists():
        raise FileNotFoundError(f"Missing live rock false-trigger seed NPZ: {seed_npz}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "passed":
            raise ValueError(f"Live rock false-trigger seed package is not passed: {summary_path}")
        if summary.get("training_use") not in {None, "accepted_overlay_derived_hard_negative"}:
            raise ValueError(f"Live rock false-trigger seed package is not accepted for training: {summary_path}")
    seed_samples: list[ThreeClassSample] = []
    with np.load(seed_npz, allow_pickle=False) as data:
        landmarks = cast(NDArray[np.float32], np.asarray(data["canonical_landmarks"], dtype=np.float32))
        mask = cast(NDArray[np.bool_], np.asarray(data["mask"], dtype=np.bool_))
        progress = cast(NDArray[np.float32], np.asarray(data["progress"], dtype=np.float32))
        lengths = cast(NDArray[np.int64], np.asarray(data["lengths"], dtype=np.int64))
        target_names = _decode_string_array(np.asarray(data["target_names"]))
        sample_ids = _decode_string_array(np.asarray(data["sample_ids"]))
        split_names = _decode_string_array(np.asarray(data["split_names"])) if "split_names" in data else []
        source_names = _decode_string_array(np.asarray(data["source_names"])) if "source_names" in data else []
        hard_flags = np.asarray(data["hard_example_flags"], dtype=np.bool_) if "hard_example_flags" in data else np.ones((landmarks.shape[0],), dtype=np.bool_)
    if landmarks.ndim != 4 or landmarks.shape[1:] != (config.sequence_length, 21, 3):
        raise ValueError(f"{seed_npz} canonical_landmarks must have shape (N,{config.sequence_length},21,3)")
    seed_count = int(landmarks.shape[0])
    if mask.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} mask must have shape (N,{config.sequence_length})")
    if progress.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} progress must have shape (N,{config.sequence_length})")
    if lengths.shape != (seed_count,):
        raise ValueError(f"{seed_npz} lengths must have shape (N,)")
    if len(target_names) != seed_count or len(sample_ids) != seed_count:
        raise ValueError(f"{seed_npz} target_names/sample_ids length mismatch")
    for index in range(seed_count):
        target_name = _target_name(target_names[index])
        if target_name != "rock":
            raise ValueError(f"{seed_npz} live rock false-trigger seeds must all be rock, got {target_name}")
        split = (
            cast(SplitName, split_names[index])
            if index < len(split_names) and split_names[index] in SPLIT_NAMES
            else split_for_index(index, seed_count, train_fraction=config.train_fraction, val_fraction=config.val_fraction)
        )
        source_name = (
            source_names[index]
            if index < len(source_names) and source_names[index]
            else "overlay_derived_live_rock_false_trigger"
        )
        seed_samples.append(
            ThreeClassSample(
                sample_id=f"live_seed_{sample_ids[index]}",
                split=split,
                target_name="rock",
                canonical_landmarks=landmarks[index].astype(np.float32, copy=True),
                mask=mask[index].astype(np.bool_, copy=True),
                progress=progress[index].astype(np.float32, copy=True),
                source_name=source_name,
                hard_example=bool(hard_flags[index]) if index < len(hard_flags) else True,
                metadata={
                    "source_seed_sample_id": sample_ids[index],
                    "seed_package_root": config.live_rock_seed_package_root.as_posix(),
                    "source_length": int(lengths[index]),
                    "augmentation_role": "rock_wait_front_fist_false_trigger_hard_negative",
                    "source_policy": "overlay-derived live rock-hold seed accepted only after quality checks.",
                },
            )
        )
    return seed_samples


def _live_rock_false_trigger_balanced_controls(
    live_rock_seed_samples: list[ThreeClassSample],
    *,
    config: ThreeClassWaitExpansionConfig,
    rng: np.random.Generator,
) -> list[ThreeClassSample]:
    """Add paper/scissors controls so live-rock hard negatives do not unbalance labels."""

    if not live_rock_seed_samples:
        return []
    control_samples: list[ThreeClassSample] = []
    for seed_index, live_seed in enumerate(live_rock_seed_samples):
        for target_name in ("paper", "scissors"):
            generated = generate_one_three_class_sample(
                target_name=cast(TargetName, target_name),
                sample_index=config.generated_per_target + 100_000 + seed_index,
                split=live_seed.split,
                rng=rng,
                sequence_length=config.sequence_length,
                min_length=config.min_length,
                augmentation_profile=config.augmentation_profile,
            )
            source_name = f"live_rock_balanced_{target_name}_control"
            control_samples.append(
                ThreeClassSample(
                    sample_id=f"live_rock_control_{target_name}_{seed_index + 1:06d}",
                    split=generated.split,
                    target_name=cast(TargetName, target_name),
                    canonical_landmarks=generated.canonical_landmarks,
                    mask=generated.mask,
                    progress=generated.progress,
                    source_name=source_name,
                    hard_example=True,
                    metadata={
                        **generated.metadata,
                        "source_seed_sample_id": live_seed.sample_id,
                        "source_seed_package_root": config.live_rock_seed_package_root.as_posix()
                        if config.live_rock_seed_package_root is not None
                        else None,
                        "augmentation_role": "balanced_control_for_live_rock_false_trigger_hard_negative",
                        "source_policy": "balanced paper/scissors controls prevent response-window collapse to wait.",
                    },
                )
            )
    return control_samples


def _copy_v7_rps_seed_samples(config: ThreeClassWaitExpansionConfig) -> list[ThreeClassSample]:
    if config.v7_seed_package_root is None:
        return []
    seed_npz = config.v7_seed_package_root / "v7_rps_seed_dataset.npz"
    seed_metadata = config.v7_seed_package_root / "seed_metadata.jsonl"
    seed_quality_summary = config.v7_seed_package_root / "seed_quality_summary.csv"
    summary_path = config.v7_seed_package_root / "seed_package_summary.json"
    required_files = (seed_npz, seed_metadata, seed_quality_summary, summary_path)
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = ", ".join(path.name for path in missing_files)
        raise FileNotFoundError(f"Missing v7 RPS seed package artifact(s): {missing_text}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError(f"v7 RPS seed package is not passed: {summary_path}")
    if summary.get("review_gate") not in {None, "manual_approved"}:
        raise ValueError(f"v7 RPS seed package is not manually approved: {summary_path}")
    seed_samples: list[ThreeClassSample] = []
    with np.load(seed_npz, allow_pickle=False) as data:
        landmarks = cast(NDArray[np.float32], np.asarray(data["canonical_landmarks"], dtype=np.float32))
        mask = cast(NDArray[np.bool_], np.asarray(data["mask"], dtype=np.bool_))
        progress = cast(NDArray[np.float32], np.asarray(data["progress"], dtype=np.float32))
        lengths = cast(NDArray[np.int64], np.asarray(data["lengths"], dtype=np.int64))
        target_names = _decode_string_array(np.asarray(data["target_names"]))
        sample_ids = _decode_string_array(np.asarray(data["sample_ids"]))
        split_names = _decode_string_array(np.asarray(data["split_names"])) if "split_names" in data else []
        source_names = _decode_string_array(np.asarray(data["source_names"])) if "source_names" in data else []
        hard_flags = np.asarray(data["hard_example_flags"], dtype=np.bool_) if "hard_example_flags" in data else np.ones((landmarks.shape[0],), dtype=np.bool_)
        source_paths = _decode_string_array(np.asarray(data["source_paths"])) if "source_paths" in data else [""] * int(landmarks.shape[0])
    if landmarks.ndim != 4 or landmarks.shape[1:] != (config.sequence_length, 21, 3):
        raise ValueError(f"{seed_npz} canonical_landmarks must have shape (N,{config.sequence_length},21,3)")
    seed_count = int(landmarks.shape[0])
    if mask.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} mask must have shape (N,{config.sequence_length})")
    if progress.shape != (seed_count, config.sequence_length):
        raise ValueError(f"{seed_npz} progress must have shape (N,{config.sequence_length})")
    if lengths.shape != (seed_count,):
        raise ValueError(f"{seed_npz} lengths must have shape (N,)")
    if len(target_names) != seed_count or len(sample_ids) != seed_count:
        raise ValueError(f"{seed_npz} target_names/sample_ids length mismatch")
    _reject_heldout_seed_paths(source_paths, seed_npz)
    for index in range(seed_count):
        target_name = _target_name(target_names[index])
        split = (
            cast(SplitName, split_names[index])
            if index < len(split_names) and split_names[index] in SPLIT_NAMES
            else split_for_index(index, seed_count, train_fraction=config.train_fraction, val_fraction=config.val_fraction)
        )
        seed_samples.append(
            ThreeClassSample(
                sample_id=f"v7_seed_{sample_ids[index]}",
                split=split,
                target_name=target_name,
                canonical_landmarks=landmarks[index].astype(np.float32, copy=True),
                mask=mask[index].astype(np.bool_, copy=True),
                progress=progress[index].astype(np.float32, copy=True),
                source_name=source_names[index] if index < len(source_names) and source_names[index] else "v7_real_rps_seed",
                hard_example=bool(hard_flags[index]) if index < len(hard_flags) else True,
                metadata={
                    "source_seed_sample_id": sample_ids[index],
                    "source_path": source_paths[index] if index < len(source_paths) else "",
                    "seed_package_root": config.v7_seed_package_root.as_posix(),
                    "source_length": int(lengths[index]),
                    "augmentation_role": "v7_reviewed_real_seed_anchor",
                    "source_policy": "manual-approved v7 RPS real skeleton segment seed",
                    "heldout_policy": "held-out 15 MP4s remain validation-only",
                },
            )
        )
    return seed_samples


def _v7_real_seed_balanced_controls(
    v7_seed_samples: list[ThreeClassSample],
    *,
    config: ThreeClassWaitExpansionConfig,
    rng: np.random.Generator,
) -> list[ThreeClassSample]:
    if not v7_seed_samples:
        return []
    counts = Counter(sample.target_name for sample in v7_seed_samples)
    target_count = max(counts.values())
    control_samples: list[ThreeClassSample] = []
    control_index = 0
    for target_name in TARGET_NAMES:
        deficit = target_count - counts.get(target_name, 0)
        for _ in range(deficit):
            control_index += 1
            generated = generate_one_three_class_sample(
                target_name=target_name,
                sample_index=config.generated_per_target + 200_000 + control_index,
                split=split_for_index(control_index - 1, max(1, target_count), train_fraction=config.train_fraction, val_fraction=config.val_fraction),
                rng=rng,
                sequence_length=config.sequence_length,
                min_length=config.min_length,
                augmentation_profile=config.augmentation_profile,
            )
            control_samples.append(
                ThreeClassSample(
                    sample_id=f"v7_seed_balance_control_{target_name}_{control_index:06d}",
                    split=generated.split,
                    target_name=target_name,
                    canonical_landmarks=generated.canonical_landmarks,
                    mask=generated.mask,
                    progress=generated.progress,
                    source_name=f"v7_real_seed_balanced_{target_name}_control",
                    hard_example=True,
                    metadata={
                        **generated.metadata,
                        "augmentation_role": "balanced_control_for_v7_real_seed_imbalance",
                        "source_seed_package_root": config.v7_seed_package_root.as_posix()
                        if config.v7_seed_package_root is not None
                        else None,
                        "source_policy": "matched procedural controls keep v7 real seeds class-balanced",
                    },
                )
            )
    return control_samples


def _arrays_for_samples(samples: list[ThreeClassSample], *, sequence_length: int) -> dict[str, NDArray[np.generic]]:
    count = len(samples)
    landmarks = np.zeros((count, sequence_length, 21, 3), dtype=np.float32)
    mask = np.zeros((count, sequence_length), dtype=np.bool_)
    lengths = np.zeros((count,), dtype=np.int64)
    progress = np.zeros((count, sequence_length), dtype=np.float32)
    target_names = np.empty((count,), dtype="<U16")
    label_names = np.empty((count,), dtype="<U24")
    split_names = np.empty((count,), dtype="<U5")
    sample_ids = np.empty((count,), dtype="<U80")
    labels = np.zeros((count,), dtype=np.int64)
    source_names = np.empty((count,), dtype="<U64")
    hard_flags = np.zeros((count,), dtype=np.bool_)
    for index, sample in enumerate(samples):
        landmarks[index] = sample.canonical_landmarks
        mask[index] = sample.mask
        lengths[index] = sample.length
        progress[index] = sample.progress
        target_names[index] = sample.target_name
        label_names[index] = sample.transition_label
        split_names[index] = sample.split
        sample_ids[index] = sample.sample_id
        labels[index] = sample.label_id
        source_names[index] = sample.source_name
        hard_flags[index] = sample.hard_example
    features = landmark_velocity_features(landmarks, mask=mask, lengths=lengths)
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "label_names": label_names,
        "target_names": target_names,
        "split_names": split_names,
        "lengths": lengths,
        "mask": mask,
        "progress": progress,
        "canonical_landmarks": landmarks,
        "features": features,
        "source_names": source_names,
        "hard_example_flags": hard_flags,
        "feature_names": np.asarray(["canonical_xyz", "velocity_xyz"]),
    }


def _start_pose(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile = "baseline",
) -> tuple[dict[FingerName, float], dict[FingerName, float]]:
    if augmentation_profile in {"v7b_rps_pose_conservative_scissors", "v7c_prompt_window_rock_guarded_paper_rescue"} and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.78, 0.99)),
            "index": float(rng.uniform(0.66, 0.99)),
            "middle": float(rng.uniform(0.66, 0.99)),
            "ring": float(rng.uniform(0.84, 0.99)),
            "pinky": float(rng.uniform(0.84, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.050)),
            "index": float(rng.normal(-0.012, 0.066)),
            "middle": float(rng.normal(0.006, 0.060)),
            "ring": float(rng.normal(0.018, 0.066)),
            "pinky": float(rng.normal(0.028, 0.072)),
        }
        return curls, spreads
    if augmentation_profile == "v4_live_prompt_hard" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.78, 0.99)),
            "index": float(rng.uniform(0.66, 0.99)),
            "middle": float(rng.uniform(0.66, 0.99)),
            "ring": float(rng.uniform(0.84, 0.99)),
            "pinky": float(rng.uniform(0.84, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.045)),
            "index": float(rng.normal(-0.010, 0.060)),
            "middle": float(rng.normal(0.006, 0.055)),
            "ring": float(rng.normal(0.016, 0.060)),
            "pinky": float(rng.normal(0.024, 0.066)),
        }
        return curls, spreads
    if augmentation_profile == "v4_hard_paper_scissors" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.80, 0.99)),
            "index": float(rng.uniform(0.72, 0.99)),
            "middle": float(rng.uniform(0.72, 0.99)),
            "ring": float(rng.uniform(0.86, 0.99)),
            "pinky": float(rng.uniform(0.86, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.038)),
            "index": float(rng.normal(-0.006, 0.045)),
            "middle": float(rng.normal(0.004, 0.042)),
            "ring": float(rng.normal(0.010, 0.045)),
            "pinky": float(rng.normal(0.016, 0.050)),
        }
        return curls, spreads
    if augmentation_profile == "v4_temporal_curl" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.80, 0.99)),
            "index": float(rng.uniform(0.72, 0.99)),
            "middle": float(rng.uniform(0.72, 0.99)),
            "ring": float(rng.uniform(0.86, 0.99)),
            "pinky": float(rng.uniform(0.86, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.038)),
            "index": float(rng.normal(-0.006, 0.045)),
            "middle": float(rng.normal(0.004, 0.042)),
            "ring": float(rng.normal(0.010, 0.045)),
            "pinky": float(rng.normal(0.016, 0.050)),
        }
        return curls, spreads
    if augmentation_profile == "v4_selector_targets" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.82, 0.99)),
            "index": float(rng.uniform(0.76, 0.99)),
            "middle": float(rng.uniform(0.76, 0.99)),
            "ring": float(rng.uniform(0.86, 0.99)),
            "pinky": float(rng.uniform(0.86, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.030)),
            "index": float(rng.normal(-0.010, 0.038)),
            "middle": float(rng.normal(0.006, 0.036)),
            "ring": float(rng.normal(0.012, 0.038)),
            "pinky": float(rng.normal(0.018, 0.040)),
        }
        return curls, spreads
    if augmentation_profile == "v4_remaining_gate" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.72, 0.99)),
            "index": float(rng.uniform(0.54, 0.94)),
            "middle": float(rng.uniform(0.54, 0.94)),
            "ring": float(rng.uniform(0.78, 0.99)),
            "pinky": float(rng.uniform(0.78, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.060)),
            "index": float(rng.normal(-0.025, 0.090)),
            "middle": float(rng.normal(0.015, 0.080)),
            "ring": float(rng.normal(0.035, 0.085)),
            "pinky": float(rng.normal(0.055, 0.095)),
        }
        return curls, spreads
    if augmentation_profile == "v4_failure_focused" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.74, 0.99)),
            "index": float(rng.uniform(0.58, 0.96)),
            "middle": float(rng.uniform(0.58, 0.96)),
            "ring": float(rng.uniform(0.80, 0.99)),
            "pinky": float(rng.uniform(0.80, 0.99)),
        }
        spreads = {
            "thumb": float(rng.normal(0.0, 0.050)),
            "index": float(rng.normal(-0.020, 0.070)),
            "middle": float(rng.normal(0.010, 0.060)),
            "ring": float(rng.normal(0.025, 0.070)),
            "pinky": float(rng.normal(0.040, 0.075)),
        }
        return curls, spreads
    if augmentation_profile == "v4_rebalanced" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.80, 0.99)),
            "index": float(rng.uniform(0.72, 0.99)),
            "middle": float(rng.uniform(0.72, 0.99)),
            "ring": float(rng.uniform(0.84, 0.99)),
            "pinky": float(rng.uniform(0.84, 0.99)),
        }
        spreads = {finger: float(rng.normal(0.0, 0.032)) for finger in FINGER_NAMES}
        return curls, spreads
    if augmentation_profile == "v4_contrastive" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.78, 0.99)),
            "index": float(rng.uniform(0.70, 0.99)),
            "middle": float(rng.uniform(0.70, 0.99)),
            "ring": float(rng.uniform(0.84, 0.99)),
            "pinky": float(rng.uniform(0.84, 0.99)),
        }
        spreads = {finger: float(rng.normal(0.0, 0.040)) for finger in FINGER_NAMES}
        return curls, spreads
    if augmentation_profile == "v4_fewshot" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.76, 0.99)),
            "index": float(rng.uniform(0.66, 0.99)),
            "middle": float(rng.uniform(0.66, 0.99)),
            "ring": float(rng.uniform(0.78, 0.99)),
            "pinky": float(rng.uniform(0.78, 0.99)),
        }
        spreads = {finger: float(rng.normal(0.0, 0.030)) for finger in FINGER_NAMES}
        return curls, spreads
    if augmentation_profile == "v3_targeted" and target_name == "rock":
        curls = {
            "thumb": float(rng.uniform(0.82, 0.99)),
            "index": float(rng.uniform(0.70, 0.98)),
            "middle": float(rng.uniform(0.70, 0.98)),
            "ring": float(rng.uniform(0.88, 0.99)),
            "pinky": float(rng.uniform(0.88, 0.99)),
        }
        spreads = {finger: float(rng.normal(0.0, 0.018)) for finger in FINGER_NAMES}
        return curls, spreads
    curls = {finger: float(rng.uniform(0.88, 0.99)) for finger in FINGER_NAMES}
    spreads = {finger: float(rng.normal(0.0, 0.015)) for finger in FINGER_NAMES}
    return curls, spreads


def _target_pose(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile = "baseline",
) -> tuple[dict[FingerName, float], dict[FingerName, float]]:
    if target_name == "rock":
        if augmentation_profile in {"v7b_rps_pose_conservative_scissors", "v7c_prompt_window_rock_guarded_paper_rescue"}:
            curls = {
                "thumb": float(rng.uniform(0.78, 0.99)),
                "index": float(rng.uniform(0.66, 0.99)),
                "middle": float(rng.uniform(0.66, 0.99)),
                "ring": float(rng.uniform(0.84, 0.99)),
                "pinky": float(rng.uniform(0.84, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.050)),
                "index": float(rng.normal(-0.012, 0.066)),
                "middle": float(rng.normal(0.006, 0.060)),
                "ring": float(rng.normal(0.018, 0.066)),
                "pinky": float(rng.normal(0.028, 0.072)),
            }
            return curls, spreads
        if augmentation_profile == "v4_prompt_wait_hard":
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.040)),
                "index": float(rng.normal(-0.010, 0.050)),
                "middle": float(rng.normal(0.006, 0.048)),
                "ring": float(rng.normal(0.014, 0.050)),
                "pinky": float(rng.normal(0.022, 0.055)),
            }
            return curls, spreads
        if augmentation_profile == "v4_paper_rescue_micro":
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.036)),
                "index": float(rng.normal(-0.008, 0.044)),
                "middle": float(rng.normal(0.004, 0.042)),
                "ring": float(rng.normal(0.012, 0.044)),
                "pinky": float(rng.normal(0.018, 0.048)),
            }
            return curls, spreads
        if augmentation_profile == "v4_final_gate_micro":
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.036)),
                "index": float(rng.normal(-0.008, 0.044)),
                "middle": float(rng.normal(0.004, 0.042)),
                "ring": float(rng.normal(0.012, 0.044)),
                "pinky": float(rng.normal(0.018, 0.048)),
            }
            return curls, spreads
        if augmentation_profile == "v4_live_prompt_hard":
            curls = {
                "thumb": float(rng.uniform(0.78, 0.99)),
                "index": float(rng.uniform(0.66, 0.99)),
                "middle": float(rng.uniform(0.66, 0.99)),
                "ring": float(rng.uniform(0.84, 0.99)),
                "pinky": float(rng.uniform(0.84, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.048)),
                "index": float(rng.normal(-0.012, 0.064)),
                "middle": float(rng.normal(0.006, 0.058)),
                "ring": float(rng.normal(0.018, 0.064)),
                "pinky": float(rng.normal(0.028, 0.070)),
            }
            return curls, spreads
        if augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"}:
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.040)),
                "index": float(rng.normal(-0.006, 0.048)),
                "middle": float(rng.normal(0.004, 0.045)),
                "ring": float(rng.normal(0.010, 0.048)),
                "pinky": float(rng.normal(0.016, 0.052)),
            }
            return curls, spreads
        if augmentation_profile == "v4_hard_paper_scissors":
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.040)),
                "index": float(rng.normal(-0.006, 0.048)),
                "middle": float(rng.normal(0.004, 0.045)),
                "ring": float(rng.normal(0.010, 0.048)),
                "pinky": float(rng.normal(0.016, 0.052)),
            }
            return curls, spreads
        if augmentation_profile == "v4_temporal_curl":
            curls = {
                "thumb": float(rng.uniform(0.80, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.040)),
                "index": float(rng.normal(-0.006, 0.048)),
                "middle": float(rng.normal(0.004, 0.045)),
                "ring": float(rng.normal(0.010, 0.048)),
                "pinky": float(rng.normal(0.016, 0.052)),
            }
            return curls, spreads
        if augmentation_profile == "v4_selector_targets":
            curls = {
                "thumb": float(rng.uniform(0.82, 0.99)),
                "index": float(rng.uniform(0.76, 0.99)),
                "middle": float(rng.uniform(0.76, 0.99)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.032)),
                "index": float(rng.normal(-0.010, 0.040)),
                "middle": float(rng.normal(0.006, 0.038)),
                "ring": float(rng.normal(0.012, 0.040)),
                "pinky": float(rng.normal(0.018, 0.042)),
            }
            return curls, spreads
        if augmentation_profile == "v4_remaining_gate":
            curls = {
                "thumb": float(rng.uniform(0.72, 0.99)),
                "index": float(rng.uniform(0.54, 0.94)),
                "middle": float(rng.uniform(0.54, 0.94)),
                "ring": float(rng.uniform(0.78, 0.99)),
                "pinky": float(rng.uniform(0.78, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.065)),
                "index": float(rng.normal(-0.025, 0.095)),
                "middle": float(rng.normal(0.015, 0.085)),
                "ring": float(rng.normal(0.035, 0.090)),
                "pinky": float(rng.normal(0.055, 0.100)),
            }
            return curls, spreads
        if augmentation_profile == "v4_failure_focused":
            curls = {
                "thumb": float(rng.uniform(0.74, 0.99)),
                "index": float(rng.uniform(0.58, 0.96)),
                "middle": float(rng.uniform(0.58, 0.96)),
                "ring": float(rng.uniform(0.80, 0.99)),
                "pinky": float(rng.uniform(0.80, 0.99)),
            }
            spreads = {
                "thumb": float(rng.normal(0.0, 0.050)),
                "index": float(rng.normal(-0.020, 0.075)),
                "middle": float(rng.normal(0.010, 0.065)),
                "ring": float(rng.normal(0.030, 0.075)),
                "pinky": float(rng.normal(0.045, 0.080)),
            }
            return curls, spreads
        if augmentation_profile == "v4_rebalanced":
            curls = {
                "thumb": float(rng.uniform(0.78, 0.99)),
                "index": float(rng.uniform(0.72, 0.99)),
                "middle": float(rng.uniform(0.72, 0.99)),
                "ring": float(rng.uniform(0.84, 0.99)),
                "pinky": float(rng.uniform(0.84, 0.99)),
            }
            spreads = {finger: float(rng.normal(0.0, 0.034)) for finger in FINGER_NAMES}
            return curls, spreads
        if augmentation_profile == "v4_contrastive":
            curls = {
                "thumb": float(rng.uniform(0.76, 0.99)),
                "index": float(rng.uniform(0.70, 0.99)),
                "middle": float(rng.uniform(0.70, 0.99)),
                "ring": float(rng.uniform(0.84, 0.99)),
                "pinky": float(rng.uniform(0.84, 0.99)),
            }
            spreads = {finger: float(rng.normal(0.0, 0.045)) for finger in FINGER_NAMES}
            return curls, spreads
        if augmentation_profile == "v4_fewshot":
            curls = {
                "thumb": float(rng.uniform(0.76, 0.99)),
                "index": float(rng.uniform(0.66, 0.99)),
                "middle": float(rng.uniform(0.66, 0.99)),
                "ring": float(rng.uniform(0.78, 0.99)),
                "pinky": float(rng.uniform(0.78, 0.99)),
            }
            spreads = {finger: float(rng.normal(0.0, 0.030)) for finger in FINGER_NAMES}
            return curls, spreads
        if augmentation_profile == "v3_targeted":
            curls = {
                "thumb": float(rng.uniform(0.82, 0.99)),
                "index": float(rng.uniform(0.72, 0.98)),
                "middle": float(rng.uniform(0.72, 0.98)),
                "ring": float(rng.uniform(0.88, 0.99)),
                "pinky": float(rng.uniform(0.88, 0.99)),
            }
            spreads = {finger: float(rng.normal(0.0, 0.018)) for finger in FINGER_NAMES}
            return curls, spreads
        curls = {finger: float(rng.uniform(0.88, 0.99)) for finger in FINGER_NAMES}
        spreads = {finger: float(rng.normal(0.0, 0.012)) for finger in FINGER_NAMES}
        return curls, spreads
    if target_name == "paper":
        if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.18)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.13)),
                "pinky": float(rng.uniform(0.01, 0.15)),
            }
            spread = float(rng.uniform(0.10, 0.50))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.50,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_prompt_wait_hard":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.18)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.08, 0.48))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.50,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_paper_rescue_micro":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.18)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.14)),
                "pinky": float(rng.uniform(0.01, 0.16)),
            }
            spread = float(rng.uniform(0.08, 0.44))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.48,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_final_gate_micro":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.20)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.10, 0.50))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.48,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_live_prompt_hard":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.20)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.12)),
                "pinky": float(rng.uniform(0.01, 0.14)),
            }
            spread = float(rng.uniform(0.10, 0.54))
            if bool(rng.integers(0, 3) == 0):
                spread = float(rng.uniform(0.06, 0.24))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.07,
                "ring": spread * 0.52,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"}:
            curls = {
                "thumb": float(rng.uniform(0.02, 0.22)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.05, 0.34))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.30,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_hard_paper_scissors":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.20)),
                "index": float(rng.uniform(0.01, 0.06)),
                "middle": float(rng.uniform(0.01, 0.07)),
                "ring": float(rng.uniform(0.01, 0.10)),
                "pinky": float(rng.uniform(0.01, 0.12)),
            }
            spread = float(rng.uniform(0.12, 0.46))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.48,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_boundary_pairs":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.22)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.10, 0.44))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.46,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_temporal_curl":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.20)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.01, 0.10)),
                "pinky": float(rng.uniform(0.01, 0.12)),
            }
            spread = float(rng.uniform(0.08, 0.38))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.42,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_selector_targets":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.22)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.01, 0.12)),
                "pinky": float(rng.uniform(0.01, 0.14)),
            }
            spread = float(rng.uniform(0.10, 0.58))
            if bool(rng.integers(0, 2)):
                spread = float(rng.uniform(0.06, 0.24))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.06,
                "ring": spread * 0.52,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_remaining_gate":
            curls = {
                "thumb": float(rng.uniform(0.01, 0.22)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.01, 0.13)),
                "pinky": float(rng.uniform(0.01, 0.15)),
            }
            spread = float(rng.uniform(0.16, 0.56))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.07,
                "ring": spread * 0.50,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_failure_focused":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.24)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.12, 0.52))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.06,
                "ring": spread * 0.52,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_rebalanced":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.22)),
                "index": float(rng.uniform(0.01, 0.09)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.01, 0.12)),
                "pinky": float(rng.uniform(0.01, 0.14)),
            }
            spread = float(rng.uniform(0.14, 0.48))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.08,
                "ring": spread * 0.48,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_contrastive":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.24)),
                "index": float(rng.uniform(0.01, 0.10)),
                "middle": float(rng.uniform(0.01, 0.10)),
                "ring": float(rng.uniform(0.01, 0.14)),
                "pinky": float(rng.uniform(0.01, 0.16)),
            }
            spread = float(rng.uniform(0.12, 0.50))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.10,
                "ring": spread * 0.45,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_fewshot":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.28)),
                "index": float(rng.uniform(0.01, 0.14)),
                "middle": float(rng.uniform(0.01, 0.14)),
                "ring": float(rng.uniform(0.01, 0.18)),
                "pinky": float(rng.uniform(0.01, 0.20)),
            }
            spread = float(rng.uniform(0.10, 0.46))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.10,
                "ring": spread * 0.45,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v3_targeted":
            curls = {
                "thumb": float(rng.uniform(0.02, 0.22)),
                "index": float(rng.uniform(0.01, 0.12)),
                "middle": float(rng.uniform(0.01, 0.12)),
                "ring": float(rng.uniform(0.01, 0.16)),
                "pinky": float(rng.uniform(0.01, 0.18)),
            }
            spread = float(rng.uniform(0.12, 0.40))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.10,
                "ring": spread * 0.40,
                "pinky": spread,
            }
            return curls, spreads
        curls = {
            "thumb": float(rng.uniform(0.04, 0.34 if augmentation_profile == "v2_targeted" else 0.30)),
            "index": float(rng.uniform(0.02, 0.20 if augmentation_profile == "v2_targeted" else 0.18)),
            "middle": float(rng.uniform(0.02, 0.20 if augmentation_profile == "v2_targeted" else 0.18)),
            "ring": float(rng.uniform(0.02, 0.24 if augmentation_profile == "v2_targeted" else 0.20)),
            "pinky": float(rng.uniform(0.02, 0.26 if augmentation_profile == "v2_targeted" else 0.22)),
        }
        spread = float(rng.uniform(0.06, 0.36 if augmentation_profile == "v2_targeted" else 0.32))
    else:
        if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.70)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_prompt_wait_hard":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.72)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.34))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_paper_rescue_micro":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.70)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_final_gate_micro":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.70)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.09)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_live_prompt_hard":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.72)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.10)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.34))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.16,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"}:
            curls = {
                "thumb": float(rng.uniform(0.18, 0.72)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.10)),
                "ring": float(rng.uniform(0.88, 0.99)),
                "pinky": float(rng.uniform(0.88, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.24,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_hard_paper_scissors":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.66)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.92, 0.99)),
                "pinky": float(rng.uniform(0.92, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_boundary_pairs":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.74)),
                "index": float(rng.uniform(0.01, 0.09)),
                "middle": float(rng.uniform(0.01, 0.10)),
                "ring": float(rng.uniform(0.88, 0.99)),
                "pinky": float(rng.uniform(0.88, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.32))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.26,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_temporal_curl":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.66)),
                "index": float(rng.uniform(0.01, 0.07)),
                "middle": float(rng.uniform(0.01, 0.08)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.30))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.24,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_selector_targets":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.74)),
                "index": float(rng.uniform(0.01, 0.10)),
                "middle": float(rng.uniform(0.01, 0.12)),
                "ring": float(rng.uniform(0.78, 0.99)),
                "pinky": float(rng.uniform(0.78, 0.99)),
            }
            spread = float(rng.uniform(0.04, 0.38))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.22,
                "ring": spread * 0.34,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_remaining_gate":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.70)),
                "index": float(rng.uniform(0.01, 0.08)),
                "middle": float(rng.uniform(0.01, 0.10)),
                "ring": float(rng.uniform(0.90, 0.99)),
                "pinky": float(rng.uniform(0.90, 0.99)),
            }
            spread = float(rng.uniform(0.04, 0.30))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.20,
                "ring": spread * 0.22,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_failure_focused":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.76)),
                "index": float(rng.uniform(0.01, 0.10)),
                "middle": float(rng.uniform(0.01, 0.12)),
                "ring": float(rng.uniform(0.88, 0.99)),
                "pinky": float(rng.uniform(0.88, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.34))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.24,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_rebalanced":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.72)),
                "index": float(rng.uniform(0.01, 0.12)),
                "middle": float(rng.uniform(0.01, 0.14)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spread = float(rng.uniform(0.07, 0.30))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.16,
                "ring": spread * 0.30,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_contrastive":
            curls = {
                "thumb": float(rng.uniform(0.18, 0.72)),
                "index": float(rng.uniform(0.01, 0.10)),
                "middle": float(rng.uniform(0.01, 0.12)),
                "ring": float(rng.uniform(0.88, 0.99)),
                "pinky": float(rng.uniform(0.88, 0.99)),
            }
            spread = float(rng.uniform(0.08, 0.34))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.18,
                "ring": spread * 0.28,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v4_fewshot":
            curls = {
                "thumb": float(rng.uniform(0.16, 0.70)),
                "index": float(rng.uniform(0.01, 0.14)),
                "middle": float(rng.uniform(0.01, 0.16)),
                "ring": float(rng.uniform(0.84, 0.99)),
                "pinky": float(rng.uniform(0.84, 0.99)),
            }
            spread = float(rng.uniform(0.05, 0.30))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.10,
                "ring": spread * 0.40,
                "pinky": spread,
            }
            return curls, spreads
        if augmentation_profile == "v3_targeted":
            curls = {
                "thumb": float(rng.uniform(0.20, 0.72)),
                "index": float(rng.uniform(0.02, 0.18)),
                "middle": float(rng.uniform(0.02, 0.18)),
                "ring": float(rng.uniform(0.86, 0.99)),
                "pinky": float(rng.uniform(0.86, 0.99)),
            }
            spread = float(rng.uniform(0.04, 0.24))
            spreads = {
                "thumb": spread * 1.2,
                "index": -spread,
                "middle": spread * 0.10,
                "ring": spread * 0.40,
                "pinky": spread,
            }
            return curls, spreads
        curls = {
            "thumb": float(rng.uniform(0.18, 0.78)),
            "index": float(rng.uniform(0.02, 0.22 if augmentation_profile == "v2_targeted" else 0.18)),
            "middle": float(rng.uniform(0.02, 0.22 if augmentation_profile == "v2_targeted" else 0.18)),
            "ring": float(rng.uniform(0.78 if augmentation_profile == "v2_targeted" else 0.82, 0.99)),
            "pinky": float(rng.uniform(0.78 if augmentation_profile == "v2_targeted" else 0.82, 0.99)),
        }
        spread = float(rng.uniform(0.03, 0.32 if augmentation_profile == "v2_targeted" else 0.26))
    spreads: dict[FingerName, float] = {
        "thumb": spread * 1.2,
        "index": -spread,
        "middle": spread * 0.10,
        "ring": spread * 0.40,
        "pinky": spread,
    }
    return curls, spreads


def _finger_onsets(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile = "baseline",
    sample_index: int = 0,
) -> dict[FingerName, float]:
    if target_name == "rock":
        return {finger: 1.0 for finger in FINGER_NAMES}
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "v7c_fast_paper_deadline_recovery":
            base = float(rng.uniform(0.00, 0.06))
            return {
                "thumb": min(0.28, base + float(rng.uniform(0.02, 0.08))),
                "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
                "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
                "ring": min(0.30, base + float(rng.uniform(0.04, 0.14))),
                "pinky": min(0.34, base + float(rng.uniform(0.06, 0.18))),
            }
        if mode == "v7c_delayed_paper_from_scissors_confuser":
            base = float(rng.uniform(0.06, 0.16))
            return {
                "thumb": min(0.38, base + float(rng.uniform(0.04, 0.12))),
                "index": min(0.26, base + float(rng.uniform(0.00, 0.06))),
                "middle": min(0.30, base + float(rng.uniform(0.02, 0.08))),
                "ring": float(rng.uniform(0.36, 0.52)),
                "pinky": float(rng.uniform(0.40, 0.52)),
            }
        base = float(rng.uniform(0.08, 0.18))
        return {
            "thumb": min(0.38, base + float(rng.uniform(0.04, 0.12))),
            "index": min(0.30, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.32, base + float(rng.uniform(0.02, 0.10))),
            "ring": float(rng.uniform(0.34, 0.48)),
            "pinky": float(rng.uniform(0.34, 0.48)),
        }
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "scissors":
        base = float(rng.uniform(0.28, 0.42))
        return {
            "thumb": min(0.52, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.50, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.54, base + float(rng.uniform(0.04, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "v7b_prompt_delayed_paper_from_scissors_confuser":
            base = float(rng.uniform(0.08, 0.20))
            return {
                "thumb": min(0.42, base + float(rng.uniform(0.04, 0.14))),
                "index": min(0.32, base + float(rng.uniform(0.00, 0.07))),
                "middle": min(0.36, base + float(rng.uniform(0.02, 0.09))),
                "ring": float(rng.uniform(0.62, 0.86)),
                "pinky": float(rng.uniform(0.66, 0.90)),
            }
        if mode == "v7b_prompt_late_ring_pinky_paper":
            base = float(rng.uniform(0.12, 0.24))
            return {
                "thumb": min(0.46, base + float(rng.uniform(0.04, 0.14))),
                "index": min(0.36, base + float(rng.uniform(0.00, 0.08))),
                "middle": min(0.40, base + float(rng.uniform(0.02, 0.10))),
                "ring": float(rng.uniform(0.54, 0.78)),
                "pinky": float(rng.uniform(0.58, 0.84)),
            }
        base = float(rng.uniform(0.10, 0.24))
        return {
            "thumb": min(0.44, base + float(rng.uniform(0.04, 0.14))),
            "index": min(0.34, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.38, base + float(rng.uniform(0.02, 0.10))),
            "ring": float(rng.uniform(0.42, 0.66)),
            "pinky": float(rng.uniform(0.46, 0.72)),
        }
    if augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "scissors":
        base = float(rng.uniform(0.20, 0.40))
        return {
            "thumb": min(0.52, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.48, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.52, base + float(rng.uniform(0.04, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_prompt_wait_hard" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "prompt_wait_paper_scissors_confuser":
            base = float(rng.uniform(0.02, 0.12))
            return {
                "thumb": min(0.34, base + float(rng.uniform(0.02, 0.10))),
                "index": min(0.22, base + float(rng.uniform(0.00, 0.05))),
                "middle": min(0.25, base + float(rng.uniform(0.00, 0.06))),
                "ring": float(rng.uniform(0.50, 0.82)),
                "pinky": float(rng.uniform(0.54, 0.88)),
            }
        if mode == "prompt_wait_paper_no_decision_recovery":
            base = float(rng.uniform(0.08, 0.22))
            return {
                "thumb": min(0.40, base + float(rng.uniform(0.02, 0.10))),
                "index": min(0.34, base + float(rng.uniform(0.00, 0.07))),
                "middle": min(0.36, base + float(rng.uniform(0.00, 0.08))),
                "ring": float(rng.uniform(0.42, 0.72)),
                "pinky": float(rng.uniform(0.46, 0.78)),
            }
        base = float(rng.uniform(0.00, 0.06))
        return {
            "thumb": min(0.28, base + float(rng.uniform(0.02, 0.08))),
            "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
            "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
            "ring": min(0.36, base + float(rng.uniform(0.04, 0.16))),
            "pinky": min(0.40, base + float(rng.uniform(0.06, 0.20))),
        }
    if augmentation_profile == "v4_prompt_wait_hard" and target_name == "scissors":
        base = float(rng.uniform(0.08, 0.24))
        return {
            "thumb": min(0.46, base + float(rng.uniform(0.04, 0.14))),
            "index": min(0.32, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.38, base + float(rng.uniform(0.02, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_paper_rescue_micro" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "paper_rescue_scissors_confuser":
            base = float(rng.uniform(0.02, 0.14))
            return {
                "thumb": min(0.36, base + float(rng.uniform(0.02, 0.12))),
                "index": min(0.24, base + float(rng.uniform(0.00, 0.06))),
                "middle": min(0.28, base + float(rng.uniform(0.00, 0.08))),
                "ring": float(rng.uniform(0.58, 0.86)),
                "pinky": float(rng.uniform(0.62, 0.90)),
            }
        base = float(rng.uniform(0.08, 0.24))
        return {
            "thumb": min(0.42, base + float(rng.uniform(0.02, 0.12))),
            "index": min(0.36, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.40, base + float(rng.uniform(0.00, 0.10))),
            "ring": float(rng.uniform(0.46, 0.74)),
            "pinky": float(rng.uniform(0.50, 0.78)),
        }
    if augmentation_profile == "v4_paper_rescue_micro" and target_name == "scissors":
        base = float(rng.uniform(0.08, 0.24))
        return {
            "thumb": min(0.44, base + float(rng.uniform(0.04, 0.14))),
            "index": min(0.32, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.38, base + float(rng.uniform(0.02, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_final_gate_micro" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "final_gate_paper_scissors_confuser":
            base = float(rng.uniform(0.00, 0.08))
            return {
                "thumb": min(0.32, base + float(rng.uniform(0.02, 0.10))),
                "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
                "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
                "ring": float(rng.uniform(0.28, 0.54)),
                "pinky": float(rng.uniform(0.32, 0.58)),
            }
        base = float(rng.uniform(0.08, 0.20))
        return {
            "thumb": min(0.36, base + float(rng.uniform(0.02, 0.11))),
            "index": min(0.28, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.30, base + float(rng.uniform(0.00, 0.07))),
            "ring": float(rng.uniform(0.24, 0.48)),
            "pinky": float(rng.uniform(0.28, 0.52)),
        }
    if augmentation_profile == "v4_final_gate_micro" and target_name == "scissors":
        base = float(rng.uniform(0.18, 0.34))
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.04, 0.14))),
            "index": min(0.42, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.46, base + float(rng.uniform(0.03, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_live_prompt_hard" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "fast_paper":
            base = float(rng.uniform(0.00, 0.06))
            return {
                "thumb": min(0.28, base + float(rng.uniform(0.02, 0.09))),
                "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
                "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
                "ring": min(0.34, base + float(rng.uniform(0.04, 0.18))),
                "pinky": min(0.38, base + float(rng.uniform(0.06, 0.22))),
            }
        if mode == "delayed_paper":
            base = float(rng.uniform(0.02, 0.12))
            return {
                "thumb": min(0.34, base + float(rng.uniform(0.02, 0.10))),
                "index": min(0.22, base + float(rng.uniform(0.00, 0.045))),
                "middle": min(0.24, base + float(rng.uniform(0.00, 0.055))),
                "ring": float(rng.uniform(0.34, 0.48)),
                "pinky": float(rng.uniform(0.38, 0.52)),
            }
        base = float(rng.uniform(0.03, 0.14))
        return {
            "thumb": min(0.36, base + float(rng.uniform(0.02, 0.11))),
            "index": min(0.24, base + float(rng.uniform(0.00, 0.055))),
            "middle": min(0.26, base + float(rng.uniform(0.00, 0.065))),
            "ring": float(rng.uniform(0.24, 0.44)),
            "pinky": float(rng.uniform(0.28, 0.50)),
        }
    if augmentation_profile == "v4_live_prompt_hard" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.18))
        return {
            "thumb": min(0.44, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.30, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.36, base + float(rng.uniform(0.02, 0.10))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_mixed_paper_timing" and target_name == "paper":
        mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
        if mode == "fast_paper":
            base = float(rng.uniform(0.00, 0.06))
            return {
                "thumb": min(0.30, base + float(rng.uniform(0.02, 0.10))),
                "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
                "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
                "ring": float(rng.uniform(0.08, 0.32)),
                "pinky": float(rng.uniform(0.10, 0.36)),
            }
        if mode == "delayed_paper":
            base = float(rng.uniform(0.02, 0.12))
            return {
                "thumb": min(0.36, base + float(rng.uniform(0.02, 0.12))),
                "index": min(0.22, base + float(rng.uniform(0.00, 0.045))),
                "middle": min(0.24, base + float(rng.uniform(0.00, 0.055))),
                "ring": float(rng.uniform(0.52, 0.78)),
                "pinky": float(rng.uniform(0.56, 0.84)),
            }
        base = float(rng.uniform(0.03, 0.14))
        return {
            "thumb": min(0.38, base + float(rng.uniform(0.02, 0.12))),
            "index": min(0.24, base + float(rng.uniform(0.00, 0.055))),
            "middle": min(0.26, base + float(rng.uniform(0.00, 0.065))),
            "ring": float(rng.uniform(0.34, 0.62)),
            "pinky": float(rng.uniform(0.38, 0.68)),
        }
    if augmentation_profile == "v4_mixed_paper_timing" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.16))
        return {
            "thumb": min(0.42, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.26, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.32, base + float(rng.uniform(0.02, 0.09))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_delayed_paper_timing" and target_name == "paper":
        base = float(rng.uniform(0.02, 0.12))
        return {
            "thumb": min(0.36, base + float(rng.uniform(0.02, 0.12))),
            "index": min(0.22, base + float(rng.uniform(0.00, 0.045))),
            "middle": min(0.24, base + float(rng.uniform(0.00, 0.055))),
            "ring": float(rng.uniform(0.50, 0.82)),
            "pinky": float(rng.uniform(0.54, 0.88)),
        }
    if augmentation_profile == "v4_delayed_paper_timing" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.16))
        return {
            "thumb": min(0.42, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.26, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.32, base + float(rng.uniform(0.02, 0.09))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_hard_paper_scissors" and target_name == "paper":
        base = float(rng.uniform(0.00, 0.06))
        return {
            "thumb": min(0.28, base + float(rng.uniform(0.02, 0.08))),
            "index": min(0.16, base + float(rng.uniform(0.00, 0.035))),
            "middle": min(0.18, base + float(rng.uniform(0.00, 0.045))),
            "ring": float(rng.uniform(0.18, 0.46)),
            "pinky": float(rng.uniform(0.22, 0.50)),
        }
    if augmentation_profile == "v4_hard_paper_scissors" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.14))
        return {
            "thumb": min(0.38, base + float(rng.uniform(0.06, 0.14))),
            "index": min(0.22, base + float(rng.uniform(0.00, 0.05))),
            "middle": min(0.26, base + float(rng.uniform(0.02, 0.07))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_boundary_pairs" and target_name == "paper":
        base = float(rng.uniform(0.00, 0.08))
        return {
            "thumb": min(0.30, base + float(rng.uniform(0.02, 0.10))),
            "index": min(0.18, base + float(rng.uniform(0.00, 0.04))),
            "middle": min(0.20, base + float(rng.uniform(0.00, 0.05))),
            "ring": float(rng.uniform(0.30, 0.62)),
            "pinky": float(rng.uniform(0.34, 0.66)),
        }
    if augmentation_profile == "v4_boundary_pairs" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.16))
        return {
            "thumb": min(0.42, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.26, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.30, base + float(rng.uniform(0.02, 0.08))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_temporal_curl" and target_name == "paper":
        base = float(rng.uniform(0.00, 0.06))
        return {
            "thumb": min(0.30, base + float(rng.uniform(0.02, 0.08))),
            "index": min(0.20, base + float(rng.uniform(0.00, 0.04))),
            "middle": min(0.22, base + float(rng.uniform(0.00, 0.05))),
            "ring": min(0.22, base + float(rng.uniform(0.00, 0.08))),
            "pinky": min(0.26, base + float(rng.uniform(0.02, 0.10))),
        }
    if augmentation_profile == "v4_temporal_curl" and target_name == "scissors":
        base = float(rng.uniform(0.04, 0.14))
        return {
            "thumb": min(0.38, base + float(rng.uniform(0.06, 0.16))),
            "index": min(0.22, base + float(rng.uniform(0.00, 0.04))),
            "middle": min(0.28, base + float(rng.uniform(0.02, 0.08))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    elif augmentation_profile == "v4_selector_targets" and target_name == "paper":
        base = float(rng.uniform(0.03, 0.16))
        return {
            "thumb": min(0.34, base + float(rng.uniform(0.02, 0.11))),
            "index": min(0.26, base + float(rng.uniform(0.00, 0.05))),
            "middle": min(0.28, base + float(rng.uniform(0.00, 0.06))),
            "ring": min(0.46, base + float(rng.uniform(0.08, 0.26))),
            "pinky": min(0.50, base + float(rng.uniform(0.10, 0.30))),
        }
    if augmentation_profile == "v4_selector_targets" and target_name == "scissors":
        base = float(rng.uniform(0.08, 0.22))
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.06, 0.18))),
            "index": min(0.34, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.40, base + float(rng.uniform(0.03, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_remaining_gate" and target_name == "paper":
        base = float(rng.uniform(0.03, 0.14))
        return {
            "thumb": min(0.36, base + float(rng.uniform(0.02, 0.12))),
            "index": min(0.28, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.30, base + float(rng.uniform(0.00, 0.07))),
            "ring": min(0.56, base + float(rng.uniform(0.16, 0.34))),
            "pinky": min(0.60, base + float(rng.uniform(0.18, 0.38))),
        }
    if augmentation_profile == "v4_remaining_gate" and target_name == "scissors":
        base = float(rng.uniform(0.08, 0.24))
        return {
            "thumb": min(0.50, base + float(rng.uniform(0.06, 0.20))),
            "index": min(0.34, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.40, base + float(rng.uniform(0.04, 0.14))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_failure_focused" and target_name == "paper":
        base = float(rng.uniform(0.06, 0.20))
        return {
            "thumb": min(0.38, base + float(rng.uniform(0.02, 0.12))),
            "index": min(0.30, base + float(rng.uniform(0.00, 0.05))),
            "middle": min(0.32, base + float(rng.uniform(0.00, 0.06))),
            "ring": min(0.46, base + float(rng.uniform(0.12, 0.24))),
            "pinky": min(0.50, base + float(rng.uniform(0.14, 0.28))),
        }
    if augmentation_profile == "v4_failure_focused" and target_name == "scissors":
        base = float(rng.uniform(0.12, 0.28))
        return {
            "thumb": min(0.52, base + float(rng.uniform(0.06, 0.18))),
            "index": min(0.36, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.42, base + float(rng.uniform(0.04, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_rebalanced" and target_name == "paper":
        base = float(rng.uniform(0.04, 0.16))
        return {
            "thumb": min(0.30, base + float(rng.uniform(0.02, 0.08))),
            "index": min(0.22, base + float(rng.uniform(0.00, 0.04))),
            "middle": min(0.22, base + float(rng.uniform(0.00, 0.04))),
            "ring": min(0.34, base + float(rng.uniform(0.04, 0.12))),
            "pinky": min(0.38, base + float(rng.uniform(0.06, 0.16))),
        }
    if augmentation_profile == "v4_rebalanced" and target_name == "scissors":
        base = float(rng.uniform(0.16, 0.30))
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.06, 0.18))),
            "index": min(0.38, base + float(rng.uniform(0.00, 0.05))),
            "middle": min(0.42, base + float(rng.uniform(0.03, 0.10))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_contrastive" and target_name == "paper":
        base = float(rng.uniform(0.12, 0.28))
        return {
            "thumb": min(0.50, base + float(rng.uniform(0.04, 0.16))),
            "index": min(0.36, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.36, base + float(rng.uniform(0.00, 0.06))),
            "ring": min(0.58, base + float(rng.uniform(0.12, 0.26))),
            "pinky": min(0.62, base + float(rng.uniform(0.14, 0.30))),
        }
    if augmentation_profile == "v4_contrastive" and target_name == "scissors":
        base = float(rng.uniform(0.18, 0.36))
        return {
            "thumb": min(0.56, base + float(rng.uniform(0.06, 0.20))),
            "index": min(0.42, base + float(rng.uniform(0.00, 0.05))),
            "middle": min(0.46, base + float(rng.uniform(0.04, 0.12))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v4_fewshot" and target_name == "paper":
        base = float(rng.uniform(0.10, 0.34))
        return {
            "thumb": min(0.52, base + float(rng.uniform(0.00, 0.16))),
            "index": min(0.42, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.42, base + float(rng.uniform(0.00, 0.08))),
            "ring": min(0.56, base + float(rng.uniform(0.06, 0.20))),
            "pinky": min(0.60, base + float(rng.uniform(0.08, 0.24))),
        }
    if augmentation_profile == "v4_fewshot" and target_name == "scissors":
        base = float(rng.uniform(0.10, 0.32))
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.04, 0.18))),
            "index": min(0.38, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.40, base + float(rng.uniform(0.00, 0.08))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v2_targeted" and target_name == "paper":
        base = float(rng.uniform(0.34, 0.58))
        return {
            "thumb": min(0.76, base + float(rng.uniform(0.02, 0.20))),
            "index": min(0.68, base + float(rng.uniform(0.00, 0.12))),
            "middle": min(0.68, base + float(rng.uniform(0.00, 0.12))),
            "ring": min(0.82, base + float(rng.uniform(0.14, 0.30))),
            "pinky": min(0.86, base + float(rng.uniform(0.16, 0.34))),
        }
    if augmentation_profile == "v2_targeted" and target_name == "scissors":
        base = float(rng.uniform(0.22, 0.42))
        return {
            "thumb": min(0.58, base + float(rng.uniform(0.06, 0.24))),
            "index": min(0.50, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.52, base + float(rng.uniform(0.00, 0.10))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    if augmentation_profile == "v3_targeted" and target_name == "paper":
        base = float(rng.uniform(0.24, 0.40))
        return {
            "thumb": min(0.56, base + float(rng.uniform(0.02, 0.14))),
            "index": min(0.48, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.48, base + float(rng.uniform(0.00, 0.08))),
            "ring": min(0.58, base + float(rng.uniform(0.06, 0.18))),
            "pinky": min(0.62, base + float(rng.uniform(0.08, 0.22))),
        }
    if augmentation_profile == "v3_targeted" and target_name == "scissors":
        base = float(rng.uniform(0.24, 0.42))
        return {
            "thumb": min(0.58, base + float(rng.uniform(0.06, 0.22))),
            "index": min(0.50, base + float(rng.uniform(0.00, 0.08))),
            "middle": min(0.52, base + float(rng.uniform(0.00, 0.10))),
            "ring": 1.0,
            "pinky": 1.0,
        }
    base = float(rng.uniform(0.24, 0.43))
    if target_name == "paper":
        return {
            "thumb": min(0.48, base + float(rng.uniform(0.00, 0.10))),
            "index": min(0.46, base + float(rng.uniform(0.00, 0.06))),
            "middle": min(0.46, base + float(rng.uniform(0.00, 0.06))),
            "ring": min(0.50, base + float(rng.uniform(0.06, 0.17))),
            "pinky": min(0.52, base + float(rng.uniform(0.08, 0.20))),
        }
    return {
        "thumb": min(0.52, base + float(rng.uniform(0.06, 0.20))),
        "index": min(0.46, base + float(rng.uniform(0.00, 0.06))),
        "middle": min(0.46, base + float(rng.uniform(0.00, 0.06))),
        "ring": 0.98,
        "pinky": 0.98,
    }


def _paper_timing_mode(
    target_name: TargetName,
    *,
    augmentation_profile: AugmentationProfile,
    sample_index: int,
) -> str | None:
    if augmentation_profile not in {
        "v4_mixed_paper_timing",
        "v4_live_prompt_hard",
        "v4_final_gate_micro",
        "v4_paper_rescue_micro",
        "v4_prompt_wait_hard",
        "v7_rps_pose",
        "v7b_rps_pose_conservative_scissors",
        "v7c_prompt_window_rock_guarded_paper_rescue",
        "v7d_real_seeded_prompt_window_guard",
        "v7e_stage1_paper_transition_rescue",
    } or target_name != "paper":
        return None
    if augmentation_profile == V7E_PROFILE:
        modes = (
            "v7e_stage1_fast_paper_transition_rescue",
            "v7e_stage1_delayed_paper_transition_rescue",
            "v7e_stage1_late_geometry_paper_transition_rescue",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == V7D_PROFILE:
        modes = (
            "v7d_real_seed_fast_paper_deadline_recovery",
            "v7d_real_seed_delayed_paper_from_scissors_confuser",
            "v7d_late_geometry_paper_rescue",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        modes = (
            "v7c_fast_paper_deadline_recovery",
            "v7c_delayed_paper_from_scissors_confuser",
            "v7c_late_geometry_paper_rescue",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        modes = (
            "v7b_prompt_delayed_paper_from_scissors_confuser",
            "v7b_prompt_late_ring_pinky_paper",
            "v7b_prompt_ambiguous_then_paper",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v7_rps_pose":
        modes = (
            "v7_hard_paper_from_live_scissors_confusion",
            "v7_fast_paper_recovery",
            "v7_delayed_ring_pinky_paper",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v4_prompt_wait_hard":
        modes = (
            "prompt_wait_paper_scissors_confuser",
            "prompt_wait_paper_no_decision_recovery",
            "prompt_wait_fast_paper_positive",
        )
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v4_paper_rescue_micro":
        modes = ("paper_rescue_scissors_confuser", "paper_rescue_no_decision")
        return modes[sample_index % len(modes)]
    if augmentation_profile == "v4_final_gate_micro":
        modes = ("final_gate_paper_scissors_confuser", "final_gate_paper_no_decision_recovery")
        return modes[sample_index % len(modes)]
    modes = ("fast_paper", "delayed_paper", "partial_ambiguous_paper")
    return modes[sample_index % len(modes)]


def _motion_progress(
    progress: float,
    *,
    onset: float,
    hesitation_center: float,
    hesitation_width: float,
    hesitation_strength: float,
) -> float:
    if progress <= onset:
        return 0.0
    normalized = max(0.0, min(1.0, (progress - onset) / max(1.0 - onset, 1.0e-6)))
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)
    hesitation = hesitation_strength * math.exp(-((progress - hesitation_center) ** 2) / (2.0 * hesitation_width * hesitation_width))
    return max(0.0, min(1.0, smooth - hesitation))


def _sample_identity(rng: np.random.Generator, *, person_id: int) -> PersonIdentity:
    handedness: Handedness = "left" if bool(rng.integers(0, 2)) else "right"
    finger_scales = {finger: float(rng.uniform(0.80, 1.24)) for finger in FINGER_NAMES}
    return PersonIdentity(
        person_id=person_id,
        handedness=handedness,
        palm_width_m=float(rng.uniform(0.073, 0.112)),
        finger_length_scales=finger_scales,
        thumb_angle_offset=float(rng.uniform(-0.20, 0.20)),
    )


def _prepare_output(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace generated files.")
    if output_root.exists():
        shards_root = output_root / "shards"
        if shards_root.exists():
            shutil.rmtree(shards_root)
        for name in (
            "sample_metadata.jsonl",
            "shard_index.csv",
            "validation_summary.json",
            "generation_config.json",
            "dataset_card.md",
            "run_summary.json",
            "training_and_validation_summary.json",
        ):
            path = output_root / name
            if path.exists():
                path.unlink()
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_config(config: ThreeClassWaitExpansionConfig) -> None:
    if config.generated_per_target <= 0:
        raise ValueError("generated_per_target must be positive")
    if config.sequence_length != 72:
        raise ValueError("sequence_length must be 72 for the current predictor contract")
    if not 1 <= config.min_length <= config.sequence_length:
        raise ValueError("min_length must be in [1, sequence_length]")
    if config.shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if config.base_rock_stride <= 0:
        raise ValueError("base_rock_stride must be positive")
    if config.augmentation_profile not in {
        "baseline",
        "v2_targeted",
        "v3_targeted",
        "v4_fewshot",
        "v4_contrastive",
        "v4_rebalanced",
        "v4_failure_focused",
        "v4_remaining_gate",
        "v4_selector_targets",
        "v4_temporal_curl",
        "v4_boundary_pairs",
        "v4_hard_paper_scissors",
        "v4_delayed_paper_timing",
        "v4_mixed_paper_timing",
        "v4_live_prompt_hard",
        "v4_final_gate_micro",
        "v4_paper_rescue_micro",
        "v4_prompt_wait_hard",
        "v7_rps_pose",
            "v7b_rps_pose_conservative_scissors",
            "v7c_prompt_window_rock_guarded_paper_rescue",
            "v7d_real_seeded_prompt_window_guard",
            "v7e_stage1_paper_transition_rescue",
        }:
            raise ValueError(
            "augmentation_profile must be baseline, v2_targeted, v3_targeted, "
            "v4_fewshot, v4_contrastive, v4_rebalanced, v4_failure_focused, "
            "v4_remaining_gate, v4_selector_targets, v4_temporal_curl, "
            "v4_boundary_pairs, v4_hard_paper_scissors, v4_delayed_paper_timing, "
            "v4_mixed_paper_timing, v4_live_prompt_hard, v4_final_gate_micro, "
            "v4_paper_rescue_micro, v4_prompt_wait_hard, v7_rps_pose, "
                "v7b_rps_pose_conservative_scissors, or "
                "v7c_prompt_window_rock_guarded_paper_rescue, or "
                "v7d_real_seeded_prompt_window_guard, or "
                "v7e_stage1_paper_transition_rescue"
            )
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 <= config.val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.train_fraction + config.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")


def _target_name(value: str) -> TargetName:
    if value not in TARGET_NAMES:
        raise ValueError(f"Unsupported target name: {value}")
    return cast(TargetName, value)


def _behavior_profile(augmentation_profile: AugmentationProfile) -> AugmentationProfile:
    if augmentation_profile in REAL_SEEDED_PROMPT_WINDOW_PROFILES:
        return V7C_BEHAVIOR_PROFILE
    return augmentation_profile


def _split_name(value: str) -> SplitName:
    if value not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split name: {value}")
    return cast(SplitName, value)


def _sample_length(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    sequence_length: int,
    min_length: int,
    augmentation_profile: AugmentationProfile,
) -> int:
    if augmentation_profile == "baseline":
        return int(rng.integers(min_length, sequence_length + 1))
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "rock":
        low = max(min_length, 46)
    elif augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        low = max(min_length, 42)
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "paper":
        low = max(min_length, 44)
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "rock":
        low = max(min_length, 46)
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors":
        low = max(min_length, 42)
    elif augmentation_profile == "v7_rps_pose" and target_name == "paper":
        low = max(min_length, 40)
    elif augmentation_profile == "v7_rps_pose" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v7_rps_pose":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_prompt_wait_hard" and target_name == "paper":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_prompt_wait_hard" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_prompt_wait_hard":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_paper_rescue_micro" and target_name == "paper":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_paper_rescue_micro" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_paper_rescue_micro":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_final_gate_micro" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_final_gate_micro" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_final_gate_micro":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_live_prompt_hard" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_live_prompt_hard" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_live_prompt_hard":
        low = max(min_length, 38)
    elif augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"} and target_name == "paper":
        low = max(min_length, 44)
    elif augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"} and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile in {"v4_delayed_paper_timing", "v4_mixed_paper_timing"}:
        low = max(min_length, 40)
    elif augmentation_profile == "v4_hard_paper_scissors" and target_name == "paper":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_hard_paper_scissors" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_hard_paper_scissors":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_boundary_pairs" and target_name == "paper":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_boundary_pairs" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_boundary_pairs":
        low = max(min_length, 40)
    if augmentation_profile == "v4_selector_targets" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_selector_targets" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_selector_targets":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_temporal_curl" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_temporal_curl" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_temporal_curl":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_remaining_gate" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_remaining_gate" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_remaining_gate":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_failure_focused" and target_name == "paper":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_failure_focused" and target_name == "rock":
        low = max(min_length, 44)
    elif augmentation_profile == "v4_failure_focused":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_rebalanced" and target_name == "paper":
        low = max(min_length, 38)
    elif augmentation_profile == "v4_rebalanced" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_rebalanced":
        low = max(min_length, 40)
    elif augmentation_profile == "v4_contrastive" and target_name == "paper":
        low = max(min_length, 46)
    elif augmentation_profile == "v4_contrastive" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_contrastive":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_fewshot" and target_name == "paper":
        low = max(min_length, 48)
    elif augmentation_profile == "v4_fewshot" and target_name == "rock":
        low = max(min_length, 42)
    elif augmentation_profile == "v4_fewshot":
        low = max(min_length, 38)
    elif augmentation_profile == "v3_targeted" and target_name == "paper":
        low = max(min_length, 54)
    elif augmentation_profile == "v3_targeted" and target_name == "rock":
        low = max(min_length, 50)
    elif augmentation_profile == "v3_targeted":
        low = max(min_length, 44)
    elif target_name == "paper":
        low = max(min_length, 58)
    elif target_name == "rock":
        low = max(min_length, 54)
    else:
        low = max(min_length, 42)
    return int(rng.integers(low, sequence_length + 1))


def _sample_viewpoint(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile,
) -> tuple[float, float, float]:
    if augmentation_profile == "baseline":
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-45.0, 45.0)), float(rng.uniform(-38.0, 38.0))
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-80.0, 80.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-80.0, 80.0))
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-80.0, 80.0)), float(rng.uniform(-78.0, 78.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-76.0, 76.0)), float(rng.uniform(-74.0, 74.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-80.0, 80.0))
    if augmentation_profile == "v7_rps_pose":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-80.0, 80.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-76.0, 76.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-86.0, 86.0)), float(rng.uniform(-88.0, 88.0))
    if augmentation_profile == "v4_prompt_wait_hard":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-80.0, 80.0)), float(rng.uniform(-78.0, 78.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-76.0, 76.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-84.0, 84.0)), float(rng.uniform(-86.0, 86.0))
    if augmentation_profile == "v4_paper_rescue_micro":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-76.0, 76.0)), float(rng.uniform(-74.0, 74.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-84.0, 84.0))
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-84.0, 84.0))
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-76.0, 76.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-84.0, 84.0))
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-70.0, 70.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-76.0, 76.0)), float(rng.uniform(-74.0, 74.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-82.0, 82.0)), float(rng.uniform(-82.0, 82.0))
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-70.0, 70.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-76.0, 76.0)), float(rng.uniform(-74.0, 74.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-80.0, 80.0)), float(rng.uniform(-80.0, 80.0))
    if augmentation_profile == "v4_selector_targets":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-72.0, 72.0)), float(rng.uniform(-70.0, 70.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-74.0, 74.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-76.0, 76.0)), float(rng.uniform(-78.0, 78.0))
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-70.0, 70.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-78.0, 78.0))
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-80.0, 80.0)), float(rng.uniform(-80.0, 80.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-72.0, 72.0)), float(rng.uniform(-70.0, 70.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-80.0, 80.0)), float(rng.uniform(-80.0, 80.0))
    if augmentation_profile == "v4_failure_focused":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-76.0, 76.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-68.0, 68.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-78.0, 78.0)), float(rng.uniform(-76.0, 76.0))
    if augmentation_profile == "v4_rebalanced":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-68.0, 68.0)), float(rng.uniform(-66.0, 66.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-62.0, 62.0)), float(rng.uniform(-58.0, 58.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-68.0, 68.0)), float(rng.uniform(-66.0, 66.0))
    if augmentation_profile == "v4_contrastive":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-68.0, 68.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-74.0, 74.0)), float(rng.uniform(-72.0, 72.0))
    if augmentation_profile == "v4_fewshot":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-70.0, 70.0)), float(rng.uniform(-66.0, 66.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-66.0, 66.0)), float(rng.uniform(-62.0, 62.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-72.0, 72.0)), float(rng.uniform(-70.0, 70.0))
    if augmentation_profile == "v3_targeted":
        if target_name == "rock":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-58.0, 58.0)), float(rng.uniform(-52.0, 52.0))
        if target_name == "paper":
            return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-60.0, 60.0)), float(rng.uniform(-54.0, 54.0))
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-54.0, 54.0)), float(rng.uniform(-46.0, 46.0))
    if target_name == "scissors":
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-64.0, 64.0)), float(rng.uniform(-58.0, 58.0))
    if target_name == "rock":
        return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-52.0, 52.0)), float(rng.uniform(-46.0, 46.0))
    return float(rng.uniform(0.0, 360.0)), float(rng.uniform(-56.0, 56.0)), float(rng.uniform(-50.0, 50.0))


def _sample_observation_noise(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile,
) -> float:
    if augmentation_profile == "baseline":
        return float(rng.uniform(0.0004, 0.0022))
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        if target_name == "rock":
            return float(rng.uniform(0.0018, 0.0072))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0058))
        return float(rng.uniform(0.0014, 0.0068))
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        if target_name == "rock":
            return float(rng.uniform(0.0018, 0.0070))
        if target_name == "paper":
            return float(rng.uniform(0.0012, 0.0062))
        return float(rng.uniform(0.0014, 0.0068))
    if augmentation_profile == "v7_rps_pose":
        if target_name == "rock":
            return float(rng.uniform(0.0016, 0.0075))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0065))
        return float(rng.uniform(0.0020, 0.0092))
    if augmentation_profile == "v4_prompt_wait_hard":
        if target_name == "rock":
            return float(rng.uniform(0.0014, 0.0068))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0060))
        return float(rng.uniform(0.0020, 0.0090))
    if augmentation_profile == "v4_paper_rescue_micro":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0058))
        if target_name == "paper":
            return float(rng.uniform(0.0009, 0.0054))
        return float(rng.uniform(0.0018, 0.0084))
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0058))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0058))
        return float(rng.uniform(0.0018, 0.0084))
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "rock":
            return float(rng.uniform(0.0014, 0.0072))
        if target_name == "paper":
            return float(rng.uniform(0.0009, 0.0062))
        return float(rng.uniform(0.0018, 0.0088))
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0058))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0056))
        return float(rng.uniform(0.0018, 0.0080))
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0058))
        if target_name == "paper":
            return float(rng.uniform(0.0012, 0.0062))
        return float(rng.uniform(0.0016, 0.0078))
    if augmentation_profile == "v4_selector_targets":
        if target_name == "rock":
            return float(rng.uniform(0.0006, 0.0032))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0075))
        return float(rng.uniform(0.0012, 0.0068))
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0058))
        if target_name == "paper":
            return float(rng.uniform(0.0009, 0.0052))
        return float(rng.uniform(0.0012, 0.0066))
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "rock":
            return float(rng.uniform(0.0016, 0.0084))
        if target_name == "paper":
            return float(rng.uniform(0.0008, 0.0056))
        return float(rng.uniform(0.0014, 0.0076))
    if augmentation_profile == "v4_failure_focused":
        if target_name == "rock":
            return float(rng.uniform(0.0014, 0.0078))
        if target_name == "paper":
            return float(rng.uniform(0.0008, 0.0050))
        return float(rng.uniform(0.0012, 0.0068))
    if augmentation_profile == "v4_rebalanced":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0060))
        if target_name == "paper":
            return float(rng.uniform(0.0007, 0.0038))
        return float(rng.uniform(0.0010, 0.0056))
    if augmentation_profile == "v4_contrastive":
        if target_name == "rock":
            return float(rng.uniform(0.0014, 0.0075))
        if target_name == "paper":
            return float(rng.uniform(0.0012, 0.0060))
        return float(rng.uniform(0.0012, 0.0068))
    if augmentation_profile == "v4_fewshot":
        if target_name == "rock":
            return float(rng.uniform(0.0010, 0.0065))
        if target_name == "paper":
            return float(rng.uniform(0.0010, 0.0058))
        return float(rng.uniform(0.0012, 0.0070))
    if augmentation_profile == "v3_targeted":
        if target_name == "scissors":
            return float(rng.uniform(0.0008, 0.0032))
        if target_name == "rock":
            return float(rng.uniform(0.0006, 0.0030))
        return float(rng.uniform(0.0008, 0.0038))
    if target_name == "scissors":
        return float(rng.uniform(0.0015, 0.0060))
    if target_name == "rock":
        return float(rng.uniform(0.0002, 0.0016))
    return float(rng.uniform(0.0006, 0.0035))


def _hesitation_params(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile,
) -> tuple[float, float, float]:
    if augmentation_profile == "baseline":
        return (
            float(rng.uniform(0.40, 0.70)),
            float(rng.uniform(0.06, 0.15)),
            float(rng.uniform(0.0, 0.18 if target_name != "rock" else 0.04)),
        )
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        if target_name == "paper":
            return float(rng.uniform(0.20, 0.52)), float(rng.uniform(0.04, 0.15)), float(rng.uniform(0.00, 0.14))
        if target_name == "rock":
            return float(rng.uniform(0.16, 0.82)), float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.00, 0.045))
        return float(rng.uniform(0.34, 0.68)), float(rng.uniform(0.05, 0.17)), float(rng.uniform(0.02, 0.14))
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        if target_name == "paper":
            return float(rng.uniform(0.32, 0.64)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.04, 0.20))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.82)), float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.0, 0.05))
        return float(rng.uniform(0.34, 0.68)), float(rng.uniform(0.05, 0.17)), float(rng.uniform(0.02, 0.14))
    if augmentation_profile == "v4_prompt_wait_hard":
        if target_name == "paper":
            return float(rng.uniform(0.24, 0.60)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.02, 0.18))
        if target_name == "rock":
            return float(rng.uniform(0.16, 0.80)), float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.0, 0.04))
        return float(rng.uniform(0.22, 0.60)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.12))
    if augmentation_profile == "v4_paper_rescue_micro":
        if target_name == "paper":
            return float(rng.uniform(0.28, 0.62)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.02, 0.18))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.76)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.22, 0.58)), float(rng.uniform(0.04, 0.15)), float(rng.uniform(0.00, 0.10))
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "paper":
            return float(rng.uniform(0.20, 0.50)), float(rng.uniform(0.04, 0.14)), float(rng.uniform(0.00, 0.12))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.76)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.24, 0.58)), float(rng.uniform(0.04, 0.15)), float(rng.uniform(0.00, 0.10))
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "paper":
            return float(rng.uniform(0.18, 0.54)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.14))
        if target_name == "rock":
            return float(rng.uniform(0.16, 0.78)), float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.0, 0.04))
        return float(rng.uniform(0.22, 0.58)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.12))
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "paper":
            return float(rng.uniform(0.18, 0.48)), float(rng.uniform(0.04, 0.13)), float(rng.uniform(0.00, 0.10))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.74)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.20, 0.54)), float(rng.uniform(0.04, 0.13)), float(rng.uniform(0.00, 0.08))
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "paper":
            return float(rng.uniform(0.22, 0.52)), float(rng.uniform(0.04, 0.14)), float(rng.uniform(0.00, 0.12))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.74)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.22, 0.56)), float(rng.uniform(0.04, 0.14)), float(rng.uniform(0.00, 0.10))
    if augmentation_profile == "v4_selector_targets":
        if target_name == "paper":
            return float(rng.uniform(0.22, 0.54)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.16))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.76)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.24, 0.62)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.12))
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "paper":
            return float(rng.uniform(0.18, 0.42)), float(rng.uniform(0.035, 0.12)), float(rng.uniform(0.00, 0.10))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.74)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.18, 0.46)), float(rng.uniform(0.035, 0.12)), float(rng.uniform(0.00, 0.08))
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "paper":
            return float(rng.uniform(0.24, 0.54)), float(rng.uniform(0.04, 0.14)), float(rng.uniform(0.00, 0.12))
        if target_name == "rock":
            return float(rng.uniform(0.16, 0.80)), float(rng.uniform(0.05, 0.20)), float(rng.uniform(0.0, 0.045))
        return float(rng.uniform(0.24, 0.60)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.11))
    if augmentation_profile == "v4_failure_focused":
        if target_name == "paper":
            return float(rng.uniform(0.30, 0.56)), float(rng.uniform(0.05, 0.14)), float(rng.uniform(0.00, 0.14))
        if target_name == "rock":
            return float(rng.uniform(0.18, 0.78)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.04))
        return float(rng.uniform(0.26, 0.62)), float(rng.uniform(0.04, 0.16)), float(rng.uniform(0.00, 0.12))
    if augmentation_profile == "v4_rebalanced":
        if target_name == "paper":
            return float(rng.uniform(0.24, 0.48)), float(rng.uniform(0.04, 0.12)), float(rng.uniform(0.00, 0.10))
        if target_name == "rock":
            return float(rng.uniform(0.20, 0.74)), float(rng.uniform(0.05, 0.16)), float(rng.uniform(0.0, 0.035))
        return float(rng.uniform(0.30, 0.60)), float(rng.uniform(0.05, 0.14)), float(rng.uniform(0.01, 0.10))
    if augmentation_profile == "v4_contrastive":
        if target_name == "paper":
            return float(rng.uniform(0.34, 0.62)), float(rng.uniform(0.06, 0.18)), float(rng.uniform(0.02, 0.20))
        if target_name == "rock":
            return float(rng.uniform(0.20, 0.76)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.04))
        return float(rng.uniform(0.32, 0.64)), float(rng.uniform(0.05, 0.15)), float(rng.uniform(0.02, 0.14))
    if augmentation_profile == "v4_fewshot":
        if target_name == "paper":
            return float(rng.uniform(0.30, 0.64)), float(rng.uniform(0.06, 0.18)), float(rng.uniform(0.00, 0.18))
        if target_name == "rock":
            return float(rng.uniform(0.20, 0.75)), float(rng.uniform(0.05, 0.18)), float(rng.uniform(0.0, 0.03))
        return float(rng.uniform(0.24, 0.58)), float(rng.uniform(0.04, 0.14)), float(rng.uniform(0.0, 0.10))
    if augmentation_profile == "v3_targeted":
        if target_name == "paper":
            return float(rng.uniform(0.42, 0.62)), float(rng.uniform(0.08, 0.18)), float(rng.uniform(0.04, 0.18))
        if target_name == "rock":
            return float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.05, 0.16)), float(rng.uniform(0.0, 0.02))
        return float(rng.uniform(0.34, 0.62)), float(rng.uniform(0.05, 0.14)), float(rng.uniform(0.0, 0.08))
    if target_name == "paper":
        return float(rng.uniform(0.50, 0.78)), float(rng.uniform(0.10, 0.24)), float(rng.uniform(0.12, 0.32))
    if target_name == "rock":
        return float(rng.uniform(0.35, 0.70)), float(rng.uniform(0.06, 0.18)), float(rng.uniform(0.0, 0.03))
    return float(rng.uniform(0.34, 0.66)), float(rng.uniform(0.05, 0.16)), float(rng.uniform(0.0, 0.10))


def _wobble_params(
    target_name: TargetName,
    rng: np.random.Generator,
    *,
    augmentation_profile: AugmentationProfile,
) -> dict[str, float]:
    if augmentation_profile == "baseline":
        return {"yaw_amp": 0.0, "pitch_amp": 0.0, "roll_amp": 0.0, "translation_std": 0.0, "phase": 0.0}
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(2.0, 10.0)),
                "pitch_amp": float(rng.uniform(2.0, 10.0)),
                "roll_amp": float(rng.uniform(2.0, 10.5)),
                "translation_std": float(rng.uniform(0.0014, 0.0082)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 8.0)),
                "pitch_amp": float(rng.uniform(1.5, 8.0)),
                "roll_amp": float(rng.uniform(1.8, 8.5)),
                "translation_std": float(rng.uniform(0.0010, 0.0068)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 8.0)),
            "pitch_amp": float(rng.uniform(2.0, 8.0)),
            "roll_amp": float(rng.uniform(3.0, 10.0)),
            "translation_std": float(rng.uniform(0.0012, 0.0074)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(2.0, 9.5)),
                "pitch_amp": float(rng.uniform(2.0, 9.5)),
                "roll_amp": float(rng.uniform(2.0, 10.0)),
                "translation_std": float(rng.uniform(0.0014, 0.0080)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.8, 8.5)),
                "pitch_amp": float(rng.uniform(1.8, 8.5)),
                "roll_amp": float(rng.uniform(2.0, 9.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0076)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 8.0)),
            "pitch_amp": float(rng.uniform(2.0, 8.0)),
            "roll_amp": float(rng.uniform(3.0, 10.0)),
            "translation_std": float(rng.uniform(0.0012, 0.0074)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_prompt_wait_hard":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(2.0, 9.0)),
                "pitch_amp": float(rng.uniform(2.0, 9.0)),
                "roll_amp": float(rng.uniform(2.0, 10.0)),
                "translation_std": float(rng.uniform(0.0014, 0.0085)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 8.0)),
                "pitch_amp": float(rng.uniform(1.5, 8.0)),
                "roll_amp": float(rng.uniform(2.0, 9.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0078)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(3.0, 12.0)),
            "pitch_amp": float(rng.uniform(3.0, 12.0)),
            "roll_amp": float(rng.uniform(8.0, 16.0)),
            "translation_std": float(rng.uniform(0.0020, 0.0110)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_paper_rescue_micro":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.2, 7.0)),
                "pitch_amp": float(rng.uniform(1.2, 7.5)),
                "roll_amp": float(rng.uniform(1.8, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0068)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(3.0, 11.0)),
            "pitch_amp": float(rng.uniform(3.0, 10.0)),
            "roll_amp": float(rng.uniform(4.0, 13.0)),
            "translation_std": float(rng.uniform(0.0018, 0.0090)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.5)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(2.0, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0068)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(3.0, 11.0)),
            "pitch_amp": float(rng.uniform(3.0, 10.0)),
            "roll_amp": float(rng.uniform(4.0, 13.0)),
            "translation_std": float(rng.uniform(0.0018, 0.0090)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(2.0, 9.0)),
                "pitch_amp": float(rng.uniform(2.0, 9.0)),
                "roll_amp": float(rng.uniform(2.0, 10.0)),
                "translation_std": float(rng.uniform(0.0014, 0.0085)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.2, 7.0)),
                "pitch_amp": float(rng.uniform(1.2, 7.5)),
                "roll_amp": float(rng.uniform(1.8, 8.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0075)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(3.0, 12.0)),
            "pitch_amp": float(rng.uniform(3.0, 12.0)),
            "roll_amp": float(rng.uniform(8.5, 16.0)),
            "translation_std": float(rng.uniform(0.0020, 0.0110)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.2, 7.2)),
                "pitch_amp": float(rng.uniform(1.2, 7.2)),
                "roll_amp": float(rng.uniform(1.8, 8.5)),
                "translation_std": float(rng.uniform(0.0010, 0.0068)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.2, 11.5)),
            "pitch_amp": float(rng.uniform(2.2, 11.5)),
            "roll_amp": float(rng.uniform(3.2, 14.5)),
            "translation_std": float(rng.uniform(0.0018, 0.0105)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 8.5)),
                "pitch_amp": float(rng.uniform(1.5, 8.5)),
                "roll_amp": float(rng.uniform(2.0, 10.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0075)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 11.0)),
            "pitch_amp": float(rng.uniform(2.0, 11.0)),
            "roll_amp": float(rng.uniform(3.0, 14.0)),
            "translation_std": float(rng.uniform(0.0016, 0.0100)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_selector_targets":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.0, 5.5)),
                "pitch_amp": float(rng.uniform(1.0, 5.5)),
                "roll_amp": float(rng.uniform(1.0, 6.0)),
                "translation_std": float(rng.uniform(0.0008, 0.0045)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(2.0, 10.0)),
                "pitch_amp": float(rng.uniform(2.0, 10.5)),
                "roll_amp": float(rng.uniform(3.0, 13.0)),
                "translation_std": float(rng.uniform(0.0020, 0.0105)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 8.5)),
            "pitch_amp": float(rng.uniform(2.0, 9.0)),
            "roll_amp": float(rng.uniform(3.0, 12.0)),
            "translation_std": float(rng.uniform(0.0014, 0.0085)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0070)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 8.0)),
                "pitch_amp": float(rng.uniform(1.5, 8.0)),
                "roll_amp": float(rng.uniform(2.0, 9.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0072)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 10.0)),
            "pitch_amp": float(rng.uniform(2.0, 10.0)),
            "roll_amp": float(rng.uniform(3.0, 13.5)),
            "translation_std": float(rng.uniform(0.0015, 0.0090)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(3.0, 12.0)),
                "pitch_amp": float(rng.uniform(3.0, 12.0)),
                "roll_amp": float(rng.uniform(3.0, 14.0)),
                "translation_std": float(rng.uniform(0.0020, 0.0120)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.0, 6.5)),
                "pitch_amp": float(rng.uniform(1.0, 7.0)),
                "roll_amp": float(rng.uniform(1.5, 7.5)),
                "translation_std": float(rng.uniform(0.0010, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 10.5)),
            "pitch_amp": float(rng.uniform(2.0, 10.5)),
            "roll_amp": float(rng.uniform(3.0, 14.0)),
            "translation_std": float(rng.uniform(0.0018, 0.0105)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_failure_focused":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(3.0, 11.0)),
                "pitch_amp": float(rng.uniform(3.0, 11.0)),
                "roll_amp": float(rng.uniform(3.0, 13.0)),
                "translation_std": float(rng.uniform(0.0020, 0.0110)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.0, 6.0)),
                "pitch_amp": float(rng.uniform(1.0, 6.5)),
                "roll_amp": float(rng.uniform(1.5, 7.0)),
                "translation_std": float(rng.uniform(0.0010, 0.0060)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 9.5)),
            "pitch_amp": float(rng.uniform(2.0, 9.5)),
            "roll_amp": float(rng.uniform(3.0, 13.0)),
            "translation_std": float(rng.uniform(0.0018, 0.0095)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_rebalanced":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(2.0, 8.0)),
                "pitch_amp": float(rng.uniform(2.0, 8.0)),
                "roll_amp": float(rng.uniform(2.0, 9.0)),
                "translation_std": float(rng.uniform(0.0015, 0.0080)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(0.8, 4.5)),
                "pitch_amp": float(rng.uniform(0.8, 4.8)),
                "roll_amp": float(rng.uniform(1.0, 5.5)),
                "translation_std": float(rng.uniform(0.0008, 0.0045)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(1.5, 7.0)),
            "pitch_amp": float(rng.uniform(1.5, 7.0)),
            "roll_amp": float(rng.uniform(2.0, 9.5)),
            "translation_std": float(rng.uniform(0.0012, 0.0070)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_contrastive":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(3.0, 10.0)),
                "pitch_amp": float(rng.uniform(3.0, 10.0)),
                "roll_amp": float(rng.uniform(3.0, 11.0)),
                "translation_std": float(rng.uniform(0.0020, 0.0100)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.0)),
                "pitch_amp": float(rng.uniform(1.5, 7.0)),
                "roll_amp": float(rng.uniform(2.0, 8.5)),
                "translation_std": float(rng.uniform(0.0015, 0.0070)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(2.0, 8.5)),
            "pitch_amp": float(rng.uniform(2.0, 8.5)),
            "roll_amp": float(rng.uniform(3.0, 12.0)),
            "translation_std": float(rng.uniform(0.0015, 0.0085)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v4_fewshot":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(1.5, 7.5)),
                "pitch_amp": float(rng.uniform(1.5, 8.0)),
                "roll_amp": float(rng.uniform(1.5, 8.5)),
                "translation_std": float(rng.uniform(0.0015, 0.0080)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(1.0, 6.0)),
                "pitch_amp": float(rng.uniform(1.0, 6.5)),
                "roll_amp": float(rng.uniform(1.5, 8.0)),
                "translation_std": float(rng.uniform(0.0012, 0.0065)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(1.5, 8.0)),
            "pitch_amp": float(rng.uniform(1.5, 8.5)),
            "roll_amp": float(rng.uniform(2.0, 10.0)),
            "translation_std": float(rng.uniform(0.0015, 0.0080)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if augmentation_profile == "v3_targeted":
        if target_name == "rock":
            return {
                "yaw_amp": float(rng.uniform(0.5, 3.5)),
                "pitch_amp": float(rng.uniform(0.5, 3.5)),
                "roll_amp": float(rng.uniform(0.5, 3.5)),
                "translation_std": float(rng.uniform(0.0008, 0.0035)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        if target_name == "paper":
            return {
                "yaw_amp": float(rng.uniform(0.6, 4.2)),
                "pitch_amp": float(rng.uniform(0.6, 4.5)),
                "roll_amp": float(rng.uniform(0.8, 5.0)),
                "translation_std": float(rng.uniform(0.0008, 0.0040)),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            }
        return {
            "yaw_amp": float(rng.uniform(0.5, 3.2)),
            "pitch_amp": float(rng.uniform(0.5, 3.5)),
            "roll_amp": float(rng.uniform(0.8, 4.2)),
            "translation_std": float(rng.uniform(0.0008, 0.0035)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if target_name == "scissors":
        return {
            "yaw_amp": float(rng.uniform(1.0, 5.0)),
            "pitch_amp": float(rng.uniform(1.0, 6.0)),
            "roll_amp": float(rng.uniform(2.5, 9.0)),
            "translation_std": float(rng.uniform(0.0015, 0.0060)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    if target_name == "rock":
        return {
            "yaw_amp": float(rng.uniform(0.2, 2.0)),
            "pitch_amp": float(rng.uniform(0.2, 2.0)),
            "roll_amp": float(rng.uniform(0.2, 2.0)),
            "translation_std": float(rng.uniform(0.0004, 0.0020)),
            "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        }
    return {
        "yaw_amp": float(rng.uniform(0.5, 3.0)),
        "pitch_amp": float(rng.uniform(0.5, 3.5)),
        "roll_amp": float(rng.uniform(0.5, 4.0)),
        "translation_std": float(rng.uniform(0.0008, 0.0035)),
        "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
    }


def _dynamic_view(
    yaw: float,
    pitch: float,
    roll: float,
    translation: NDArray[np.float32],
    *,
    frame_index: int,
    frame_progress: float,
    rng: np.random.Generator,
    wobble: dict[str, float],
) -> tuple[float, float, float, NDArray[np.float32]]:
    phase = float(wobble["phase"]) + frame_progress * 2.0 * math.pi
    dynamic_translation = translation + cast(
        NDArray[np.float32],
        rng.normal(0.0, float(wobble["translation_std"]), size=(3,)).astype(np.float32),
    )
    return (
        yaw + float(wobble["yaw_amp"]) * math.sin(phase + frame_index * 0.05),
        pitch + float(wobble["pitch_amp"]) * math.sin(phase * 0.7 + 0.4),
        roll + float(wobble["roll_amp"]) * math.sin(phase * 1.2 + 0.8),
        dynamic_translation,
    )


def _apply_canonical_observation_noise(
    canonical: NDArray[np.float32],
    *,
    target_name: TargetName,
    frame_progress: float,
    rng: np.random.Generator,
    augmentation_profile: AugmentationProfile,
) -> NDArray[np.float32]:
    if augmentation_profile == "baseline":
        return canonical
    noisy = canonical.astype(np.float32, copy=True)
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "rock":
        sigma = 0.0028 + 0.0056 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "paper":
        sigma = 0.0014 + 0.0032 * max(0.0, 0.52 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.5, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.2, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.55 * min(1.0, max(0.0, frame_progress - 0.18) / 0.34)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue" and target_name == "scissors":
        sigma = 0.0018 + 0.0048 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.45, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.50)
        opening_gain = 0.92 + 0.24 * min(1.0, max(0.0, frame_progress - 0.28) / 0.42)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "rock":
        sigma = 0.0026 + 0.0052 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.9, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "paper":
        sigma = 0.0016 + 0.0038 * max(0.0, 0.62 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.7, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.6, size=(3,)).astype(np.float32))
        ring_pinky_gain = 0.88 + 0.72 * min(1.0, max(0.0, frame_progress - 0.36) / 0.42)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v7b_rps_pose_conservative_scissors" and target_name == "scissors":
        sigma = 0.0018 + 0.0048 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.2, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.45, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.50)
        opening_gain = 0.92 + 0.24 * min(1.0, max(0.0, frame_progress - 0.28) / 0.42)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_prompt_wait_hard" and target_name == "rock":
        sigma = 0.0024 + 0.0054 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.1, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_prompt_wait_hard" and target_name == "paper":
        sigma = 0.0015 + 0.0036 * max(0.0, 0.56 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.4, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.48 * min(1.0, max(0.0, frame_progress - 0.24) / 0.38)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v4_prompt_wait_hard" and target_name == "scissors":
        sigma = 0.0024 + 0.0060 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 3.0, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.55, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.56)
        opening_gain = 1.0 + 0.14 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_paper_rescue_micro" and target_name == "rock":
        sigma = 0.0018 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.8, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_paper_rescue_micro" and target_name == "paper":
        sigma = 0.0012 + 0.0028 * max(0.0, 0.60 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.4, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.9, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.35 * min(1.0, max(0.0, frame_progress - 0.42) / 0.35)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v4_paper_rescue_micro" and target_name == "scissors":
        sigma = 0.0020 + 0.0054 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.7, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.55, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.58)
        opening_gain = 1.0 + 0.10 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_final_gate_micro" and target_name == "rock":
        sigma = 0.0018 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.8, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_final_gate_micro" and target_name == "paper":
        sigma = 0.0014 + 0.0032 * max(0.0, 0.52 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.5, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.2, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.45 * min(1.0, max(0.0, frame_progress - 0.20) / 0.32)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v4_final_gate_micro" and target_name == "scissors":
        sigma = 0.0020 + 0.0054 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.7, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.55, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.58)
    elif augmentation_profile == "v4_live_prompt_hard" and target_name == "rock":
        sigma = 0.0024 + 0.0050 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_live_prompt_hard" and target_name == "paper":
        sigma = 0.0015 + 0.0034 * max(0.0, 0.52 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.5, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.4, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.50 * min(1.0, max(0.0, frame_progress - 0.18) / 0.34)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v4_live_prompt_hard" and target_name == "scissors":
        sigma = 0.0022 + 0.0058 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 3.0, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.55, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.56)
        opening_gain = 1.0 + 0.14 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_hard_paper_scissors" and target_name == "rock":
        sigma = 0.0018 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_hard_paper_scissors" and target_name == "paper":
        sigma = 0.0014 + 0.0028 * max(0.0, 0.48 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.5, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.7, size=(3,)).astype(np.float32))
        ring_pinky_gain = 1.0 + 0.45 * min(1.0, max(0.0, frame_progress - 0.18) / 0.32)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(ring_pinky_gain)
    elif augmentation_profile == "v4_hard_paper_scissors" and target_name == "scissors":
        sigma = 0.0020 + 0.0052 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.9, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.50, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.58)
        opening_gain = 1.0 + 0.12 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_boundary_pairs" and target_name == "rock":
        sigma = 0.0018 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_boundary_pairs" and target_name == "paper":
        sigma = 0.0016 + 0.0030 * max(0.0, 0.50 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.7, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.2, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_boundary_pairs" and target_name == "scissors":
        sigma = 0.0020 + 0.0050 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.8, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.65, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.66)
    elif augmentation_profile == "v4_temporal_curl" and target_name == "rock":
        sigma = 0.0018 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.7, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_temporal_curl" and target_name == "paper":
        sigma = 0.0014 + 0.0028 * max(0.0, 0.48 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.8, size=(3,)).astype(np.float32))
        extension_gain = 1.0 + 0.35 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(extension_gain)
    elif augmentation_profile == "v4_temporal_curl" and target_name == "scissors":
        sigma = 0.0018 + 0.0048 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.8, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.55, size=(3,)).astype(np.float32))
        for mcp, joints in ((13, (14, 15, 16)), (17, (18, 19, 20))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(0.62)
        opening_gain = 1.0 + 0.10 * min(1.0, frame_progress / 0.50)
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12))):
            base = noisy[mcp].copy()
            for joint in joints:
                noisy[joint] = base + (noisy[joint] - base) * np.float32(opening_gain)
    elif augmentation_profile == "v4_selector_targets" and target_name == "rock":
        sigma = 0.0010 + 0.0024 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_selector_targets" and target_name == "paper":
        sigma = 0.0020 + 0.0045 * max(0.0, 0.55 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.3, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_selector_targets" and target_name == "scissors":
        sigma = 0.0020 + 0.0042 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.4, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.2, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_remaining_gate" and target_name == "rock":
        sigma = 0.0032 + 0.0068 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.7, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_remaining_gate" and target_name == "paper":
        sigma = 0.0013 + 0.0028 * max(0.0, 0.50 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.4, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_remaining_gate" and target_name == "scissors":
        sigma = 0.0020 + 0.0052 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.8, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.7, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_failure_focused" and target_name == "rock":
        sigma = 0.0030 + 0.0060 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.4, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_failure_focused" and target_name == "paper":
        sigma = 0.0015 + 0.0030 * max(0.0, 0.52 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.5, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.1, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_failure_focused" and target_name == "scissors":
        sigma = 0.0020 + 0.0048 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.6, size=(3,)).astype(np.float32))
        for index in (16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 0.8, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_rebalanced" and target_name == "rock":
        sigma = 0.0022 + 0.0038 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.9, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_rebalanced" and target_name == "paper":
        sigma = 0.0012 + 0.0020 * max(0.0, 0.42 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12, 16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.3, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_rebalanced" and target_name == "scissors":
        sigma = 0.0018 + 0.0032 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_contrastive" and target_name == "rock":
        sigma = 0.0030 + 0.0050 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.5, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_contrastive" and target_name == "paper":
        sigma = 0.0024 + 0.0036 * max(0.0, 0.58 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12, 16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.9, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_contrastive" and target_name == "scissors":
        sigma = 0.0022 + 0.0042 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.4, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_fewshot" and target_name == "rock":
        sigma = 0.0025 + 0.0040 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.0, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_fewshot" and target_name == "paper":
        sigma = 0.0020 + 0.0030 * max(0.0, 0.55 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12, 16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.8, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v4_fewshot" and target_name == "scissors":
        sigma = 0.0024 + 0.0044 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 2.2, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v3_targeted" and target_name == "rock":
        sigma = 0.0014 + 0.0014 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (5, 6, 7, 8, 9, 10, 11, 12):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.4, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v3_targeted" and target_name == "paper":
        sigma = 0.0018 + 0.0024 * max(0.0, 0.62 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (13, 14, 15, 16, 17, 18, 19, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.6, size=(3,)).astype(np.float32))
    elif augmentation_profile == "v3_targeted" and target_name == "scissors":
        sigma = 0.0015 + 0.0016 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
    elif target_name == "scissors":
        sigma = 0.0035 + 0.0030 * math.sin(frame_progress * math.pi) ** 2
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
        for index in (8, 12, 16, 20):
            noisy[index] += cast(NDArray[np.float32], rng.normal(0.0, sigma * 1.7, size=(3,)).astype(np.float32))
    elif target_name == "paper":
        sigma = 0.0018 + 0.0020 * max(0.0, 0.65 - frame_progress)
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
    else:
        sigma = 0.0008
        noisy += cast(NDArray[np.float32], rng.normal(0.0, sigma, size=noisy.shape).astype(np.float32))
    noisy -= noisy[0].copy()
    return noisy


def _source_name(target_name: TargetName, augmentation_profile: AugmentationProfile, *, sample_index: int = 0) -> str:
    if augmentation_profile == "baseline":
        return "three_class_procedural_3d"
    if augmentation_profile == V7E_PROFILE:
        if target_name == "rock":
            return "v7e_stage1_rock_wait_prompt_guard"
        if target_name == "paper":
            return "v7e_stage1_paper_transition_rescue"
        return "v7e_stage1_delayed_scissors_boundary_control"
    if augmentation_profile == V7D_PROFILE:
        if target_name == "rock":
            return "v7d_real_seeded_rock_wait_prompt_guard"
        if target_name == "paper":
            mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
            if mode == "v7d_real_seed_fast_paper_deadline_recovery":
                return "v7d_real_seed_fast_paper_deadline_recovery"
            if mode == "v7d_real_seed_delayed_paper_from_scissors_confuser":
                return "v7d_real_seed_delayed_paper_from_scissors_confuser"
            return "v7d_real_seed_late_geometry_paper_rescue"
        return "v7d_real_seeded_delayed_scissors_boundary_control"
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        if target_name == "rock":
            return "v7c_prompt_window_rock_guard_hard_negative"
        if target_name == "paper":
            mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
            if mode == "v7c_fast_paper_deadline_recovery":
                return "v7c_fast_paper_deadline_recovery"
            if mode == "v7c_delayed_paper_from_scissors_confuser":
                return "v7c_delayed_paper_from_scissors_confuser"
            return "v7c_late_geometry_paper_rescue"
        return "v7c_delayed_scissors_control_preserved"
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        if target_name == "rock":
            return "v7b_rps_pose_rock_wait_prompt_hard_negative"
        if target_name == "paper":
            return "v7b_rps_pose_delayed_paper_scissors_confuser"
        return "v7b_rps_pose_delayed_scissors_control"
    if augmentation_profile == "v7_rps_pose":
        if target_name == "rock":
            return "v7_rps_pose_rock_wait_false_trigger_hard_negative"
        if target_name == "paper":
            return "v7_rps_pose_paper_vs_scissors_confusion_recovery"
        return "v7_rps_pose_scissors_transition_and_static_variation"
    if augmentation_profile == "v4_prompt_wait_hard":
        if target_name == "rock":
            return "v4_prompt_wait_hard_rock_wait_prompt_negative"
        if target_name == "paper":
            mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
            if mode == "prompt_wait_paper_scissors_confuser":
                return "v4_prompt_wait_hard_paper_from_scissors_confuser"
            if mode == "prompt_wait_paper_no_decision_recovery":
                return "v4_prompt_wait_hard_paper_from_no_decision"
            return "v4_prompt_wait_hard_fast_paper_positive"
        return "v4_prompt_wait_hard_shaky_scissors_control"
    if augmentation_profile == "v4_paper_rescue_micro":
        if target_name == "rock":
            return "v4_paper_rescue_micro_rock_wait_control"
        if target_name == "paper":
            mode = _paper_timing_mode(target_name, augmentation_profile=augmentation_profile, sample_index=sample_index)
            if mode == "paper_rescue_scissors_confuser":
                return "v4_paper_rescue_micro_hard_paper_from_scissors_confuser"
            return "v4_paper_rescue_micro_hard_paper_from_no_decision"
        return "v4_paper_rescue_micro_shaky_scissors_control"
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "rock":
            return "v4_final_gate_micro_rock_wait_control"
        if target_name == "paper":
            return "v4_final_gate_micro_paper_boundary_recovery"
        return "v4_final_gate_micro_late_scissors_recovery"
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "rock":
            return "v4_live_prompt_hard_rock_wait_prompt_control"
        if target_name == "paper":
            return "v4_live_prompt_hard_fast_and_delayed_paper_positive"
        return "v4_live_prompt_hard_shaky_scissors_positive"
    if augmentation_profile == "v4_mixed_paper_timing":
        if target_name == "rock":
            return "v4_mixed_paper_timing_rock_wait_control"
        if target_name == "paper":
            return "v4_mixed_paper_timing_mixture_paper"
        return "v4_mixed_paper_timing_shaky_scissors_control"
    if augmentation_profile == "v4_delayed_paper_timing":
        if target_name == "rock":
            return "v4_delayed_paper_timing_rock_wait_control"
        if target_name == "paper":
            return "v4_delayed_paper_timing_late_ring_pinky_paper"
        return "v4_delayed_paper_timing_shaky_scissors_control"
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "rock":
            return "v4_hard_paper_scissors_rock_wait_control"
        if target_name == "paper":
            return "v4_hard_paper_scissors_late_ring_pinky_paper"
        return "v4_hard_paper_scissors_shaky_scissors_positive"
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "rock":
            return "v4_boundary_pairs_rock_wait_control"
        if target_name == "paper":
            return "v4_boundary_pairs_hard_paper_vs_scissors"
        return "v4_boundary_pairs_shaky_scissors_vs_paper"
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "rock":
            return "v4_temporal_curl_rock_wait_control"
        if target_name == "paper":
            return "v4_temporal_curl_matched_hard_paper_control"
        return "v4_temporal_curl_early_scissors_positive"
    if augmentation_profile == "v4_selector_targets":
        if target_name == "rock":
            return "v4_selector_rock_wait_false_transition_hard_negative"
        if target_name == "paper":
            return "v4_selector_paper_scissors_boundary_and_unstable_positive"
        return "v4_selector_scissors_paper_boundary_hard_positive"
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "rock":
            return "v4_remaining_gate_rock_false_paper_scissors_negative"
        if target_name == "paper":
            return "v4_remaining_gate_early_paper_vs_scissors_boundary"
        return "v4_remaining_gate_shaky_scissors_vs_paper_boundary"
    if augmentation_profile == "v4_failure_focused":
        if target_name == "rock":
            return "v4_failure_focused_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_failure_focused_paper_scissors_confuser"
        return "v4_failure_focused_late_scissors_recovery"
    if augmentation_profile == "v4_rebalanced":
        if target_name == "rock":
            return "v4_rebalanced_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_rebalanced_fast_paper_recovery"
        return "v4_rebalanced_scissors_boundary_control"
    if augmentation_profile == "v4_contrastive":
        if target_name == "rock":
            return "v4_contrastive_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_contrastive_early_paper_boundary"
        return "v4_contrastive_delayed_scissors_boundary"
    if augmentation_profile == "v4_fewshot":
        if target_name == "rock":
            return "v4_fewshot_rock_wait_jitter"
        if target_name == "paper":
            return "v4_fewshot_fast_and_delayed_paper"
        return "v4_fewshot_shaky_view_scissors"
    if augmentation_profile == "v3_targeted":
        if target_name == "rock":
            return "v3_anti_scissors_rock_wait"
        if target_name == "paper":
            return "v3_anti_scissors_paper"
        return "v3_control_scissors"
    if target_name == "rock":
        return "v2_rock_wait_hold"
    if target_name == "paper":
        return "v2_slow_fist_like_paper"
    return "v2_shaky_view_scissors"


def _hard_case_name(
    target_name: TargetName,
    *,
    augmentation_profile: AugmentationProfile = "baseline",
    sample_index: int = 0,
) -> str:
    if augmentation_profile == V7E_PROFILE:
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == V7D_PROFILE:
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v7c_prompt_window_rock_guarded_paper_rescue":
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v7b_rps_pose_conservative_scissors":
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v7_rps_pose":
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v4_prompt_wait_hard":
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v4_paper_rescue_micro":
        return _source_name(target_name, augmentation_profile, sample_index=sample_index)
    if augmentation_profile == "v4_final_gate_micro":
        if target_name == "rock":
            return "v4_final_gate_micro_rock_wait_control"
        if target_name == "paper":
            return "v4_final_gate_micro_paper_boundary_recovery"
        return "v4_final_gate_micro_late_scissors_recovery"
    if augmentation_profile == "v4_live_prompt_hard":
        if target_name == "rock":
            return "v4_live_prompt_hard_rock_wait_prompt_control"
        if target_name == "paper":
            return "v4_live_prompt_hard_fast_and_delayed_paper_positive"
        return "v4_live_prompt_hard_shaky_scissors_positive"
    if augmentation_profile == "v4_mixed_paper_timing":
        if target_name == "rock":
            return "v4_mixed_paper_timing_rock_wait_control"
        if target_name == "paper":
            return "v4_mixed_paper_timing_mixture_paper"
        return "v4_mixed_paper_timing_shaky_scissors_control"
    if augmentation_profile == "v4_delayed_paper_timing":
        if target_name == "rock":
            return "v4_delayed_paper_timing_rock_wait_control"
        if target_name == "paper":
            return "v4_delayed_paper_timing_late_ring_pinky_paper"
        return "v4_delayed_paper_timing_shaky_scissors_control"
    if augmentation_profile == "v4_hard_paper_scissors":
        if target_name == "rock":
            return "v4_hard_paper_scissors_rock_wait_control"
        if target_name == "paper":
            return "v4_hard_paper_scissors_late_ring_pinky_paper"
        return "v4_hard_paper_scissors_shaky_scissors_positive"
    if augmentation_profile == "v4_boundary_pairs":
        if target_name == "rock":
            return "v4_boundary_pairs_rock_wait_control"
        if target_name == "paper":
            return "v4_boundary_pairs_hard_paper_vs_scissors"
        return "v4_boundary_pairs_shaky_scissors_vs_paper"
    if augmentation_profile == "v4_temporal_curl":
        if target_name == "rock":
            return "v4_temporal_curl_rock_wait_control"
        if target_name == "paper":
            return "v4_temporal_curl_matched_hard_paper_control"
        return "v4_temporal_curl_early_scissors_positive"
    if augmentation_profile == "v4_selector_targets":
        if target_name == "rock":
            return "v4_selector_rock_wait_false_transition_hard_negative"
        if target_name == "paper":
            return "v4_selector_paper_scissors_boundary_and_unstable_positive"
        return "v4_selector_scissors_paper_boundary_hard_positive"
    if augmentation_profile == "v4_remaining_gate":
        if target_name == "rock":
            return "v4_remaining_gate_rock_false_paper_scissors_negative"
        if target_name == "paper":
            return "v4_remaining_gate_early_paper_vs_scissors_boundary"
        return "v4_remaining_gate_shaky_scissors_vs_paper_boundary"
    if augmentation_profile == "v4_failure_focused":
        if target_name == "rock":
            return "v4_failure_focused_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_failure_focused_paper_scissors_confuser"
        return "v4_failure_focused_late_scissors_recovery"
    if augmentation_profile == "v4_rebalanced":
        if target_name == "rock":
            return "v4_rebalanced_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_rebalanced_fast_paper_recovery"
        return "v4_rebalanced_scissors_boundary_control"
    if augmentation_profile == "v4_contrastive":
        if target_name == "rock":
            return "v4_contrastive_rock_false_trigger_negative"
        if target_name == "paper":
            return "v4_contrastive_early_paper_boundary"
        return "v4_contrastive_delayed_scissors_boundary"
    if augmentation_profile == "v4_fewshot":
        if target_name == "rock":
            return "v4_fewshot_rock_wait_jitter"
        if target_name == "paper":
            return "v4_fewshot_fast_and_delayed_paper"
        return "v4_fewshot_shaky_view_scissors"
    if augmentation_profile == "v3_targeted":
        if target_name == "rock":
            return "v3_anti_scissors_rock_wait"
        if target_name == "paper":
            return "v3_anti_scissors_paper"
        return "v3_control_scissors"
    if augmentation_profile == "v2_targeted":
        if target_name == "rock":
            return "v2_rock_wait_counter_paper"
        if target_name == "paper":
            return "v2_slow_fist_like_paper"
        return "v2_shaky_viewpoint_scissors"
    if target_name == "rock":
        return "rock_hold_wait_counter_paper"
    if target_name == "paper":
        return "slow_fist_like_paper"
    return "changed_viewpoint_scissors_control"


def _decode_string_array(values: NDArray[np.generic]) -> list[str]:
    return [str(value) for value in values.tolist()]


def _reject_heldout_seed_paths(source_paths: Sequence[str], seed_npz: Path) -> None:
    for source_path in source_paths:
        normalized = source_path.replace("\\", "/").lower()
        if "/test/" in normalized or normalized.endswith("/test"):
            raise ValueError(f"{seed_npz} contains held-out test source path: {source_path}")


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    return value


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "TARGET_NAMES",
    "ThreeClassSample",
    "ThreeClassWaitExpansionConfig",
    "generate_one_three_class_sample",
    "generate_three_class_hard_samples",
    "generate_three_class_wait_dataset",
    "validate_three_class_wait_dataset",
]
