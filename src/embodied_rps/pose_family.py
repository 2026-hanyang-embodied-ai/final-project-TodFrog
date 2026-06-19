"""Robust multi-person RPS pose-family skeleton generation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import yaml
from numpy.typing import NDArray

FingerName: TypeAlias = Literal["thumb", "index", "middle", "ring", "pinky"]
ClearRpsLabel: TypeAlias = Literal["rock", "paper", "scissors"]
PoseFamilyLabel: TypeAlias = Literal["rock", "paper", "scissors", "ambiguous"]
Handedness: TypeAlias = Literal["right", "left"]
StartPose: TypeAlias = Literal["neutral", "fist"]
MotionStyle: TypeAlias = Literal["smoothstep", "staggered_finger_open"]

FINGER_NAMES: tuple[FingerName, ...] = ("thumb", "index", "middle", "ring", "pinky")
CLASSIFIER_FINGERS: tuple[FingerName, ...] = ("index", "middle", "ring", "pinky")
CLEAR_RPS_LABELS: tuple[ClearRpsLabel, ...] = ("rock", "paper", "scissors")
POSE_FAMILY_LABELS: tuple[PoseFamilyLabel, ...] = ("rock", "paper", "scissors", "ambiguous")
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
KEYPOINT_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
_KEYPOINT_INDEX = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
TIP_INDICES: tuple[int, ...] = (
    _KEYPOINT_INDEX["thumb_tip"],
    _KEYPOINT_INDEX["index_tip"],
    _KEYPOINT_INDEX["middle_tip"],
    _KEYPOINT_INDEX["ring_tip"],
    _KEYPOINT_INDEX["pinky_tip"],
)


@dataclass(frozen=True)
class PoseFamilyConfig:
    """Configuration for robust synthetic pose-family generation."""

    seed: int
    person_count: int
    samples_per_label_per_person: int
    sequence_length: int
    train_fraction: float
    val_fraction: float
    labels: tuple[PoseFamilyLabel, ...]
    output_path: Path
    metadata_path: Path
    audit_path: Path
    yaw_degrees: tuple[float, ...]
    pitch_degrees: tuple[float, ...]
    val_yaw_degrees: tuple[float, ...]
    test_yaw_degrees: tuple[float, ...]
    handedness_options: tuple[Handedness, ...]
    palm_width_range: tuple[float, float]
    finger_length_scale_range: tuple[float, float]
    speed_range_s: tuple[float, float]
    start_pose: StartPose
    start_hold_ratio: float
    transition_onset_range: tuple[float, float]
    finger_open_delay_range: tuple[float, float]
    motion_style: MotionStyle
    abstain_before_motion: bool
    hesitation_probability: float
    noise_std: float
    family_ranges: dict[PoseFamilyLabel, dict[str, tuple[float, float]]]
    extended_max: float
    flexed_min: float


@dataclass(frozen=True)
class PersonIdentity:
    """Synthetic person-level hand-shape parameters."""

    person_id: int
    handedness: Handedness
    palm_width_m: float
    finger_length_scales: dict[FingerName, float]
    thumb_angle_offset: float


@dataclass(frozen=True)
class PoseFamilySample:
    """One generated trajectory sample."""

    label: PoseFamilyLabel
    person_id: int
    view_id: int
    handedness: Handedness
    yaw_degrees: float
    pitch_degrees: float
    keypoints: NDArray[np.float32]
    canonical_keypoints: NDArray[np.float32]
    start_finger_curls: dict[FingerName, float]
    start_finger_spreads: dict[FingerName, float]
    final_finger_curls: dict[FingerName, float]
    final_finger_spreads: dict[FingerName, float]


@dataclass(frozen=True)
class PoseFamilyDataset:
    """Saved pose-family tensors and metadata."""

    sequences: NDArray[np.float32]
    positions: NDArray[np.float32]
    velocities: NDArray[np.float32]
    keypoints: NDArray[np.float32]
    canonical_keypoints: NDArray[np.float32]
    start_finger_curls: NDArray[np.float32]
    final_finger_curls: NDArray[np.float32]
    labels: NDArray[np.int64]
    splits: NDArray[np.int64]
    identity_splits: NDArray[np.int64]
    view_splits: NDArray[np.int64]
    person_ids: NDArray[np.int64]
    view_ids: NDArray[np.int64]
    yaw_values: NDArray[np.float32]
    pitch_values: NDArray[np.float32]
    label_names: tuple[PoseFamilyLabel, ...]
    split_names: tuple[str, ...]
    keypoint_names: tuple[str, ...]


def load_pose_family_config(path: Path) -> PoseFamilyConfig:
    """Load and validate a pose-family YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        loaded: object = yaml.safe_load(handle)
    root = _as_mapping(loaded, "pose-family config")
    output = _as_mapping(_required(root, "output"), "output")
    views = _as_mapping(_required(root, "views"), "views")
    person_variation = _as_mapping(_required(root, "person_variation"), "person_variation")
    motion = _as_mapping(_required(root, "motion"), "motion")
    families_root = _as_mapping(_required(root, "families"), "families")
    ambiguity = _as_mapping(_required(root, "ambiguity"), "ambiguity")

    train_fraction = _fraction(_required_float(root, "train_fraction"), "train_fraction")
    val_fraction = _fraction(_required_float(root, "val_fraction"), "val_fraction")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.0")

    labels = _label_tuple(_required(root, "labels"), "labels")
    family_ranges: dict[PoseFamilyLabel, dict[str, tuple[float, float]]] = {}
    for label in labels:
        ranges = _as_mapping(_required(families_root, label), f"families.{label}")
        parsed_ranges: dict[str, tuple[float, float]] = {}
        for key in FINGER_NAMES:
            parsed_ranges[key] = _range_pair(_required(ranges, key), f"families.{label}.{key}")
        parsed_ranges["spread"] = _ordered_range_pair(_required(ranges, "spread"), f"families.{label}.spread")
        family_ranges[label] = parsed_ranges

    return PoseFamilyConfig(
        seed=_required_int(root, "seed"),
        person_count=_positive_int(_required_int(root, "person_count"), "person_count"),
        samples_per_label_per_person=_positive_int(_required_int(root, "samples_per_label_per_person"), "samples_per_label_per_person"),
        sequence_length=_positive_int(_required_int(root, "sequence_length"), "sequence_length"),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        labels=labels,
        output_path=Path(_required_string(output, "dataset_npz")),
        metadata_path=Path(_required_string(output, "metadata_json")),
        audit_path=Path(_required_string(output, "audit_json")),
        yaw_degrees=_float_tuple(_required(views, "yaw_degrees"), "views.yaw_degrees"),
        pitch_degrees=_float_tuple(_required(views, "pitch_degrees"), "views.pitch_degrees"),
        val_yaw_degrees=_float_tuple(_required(views, "val_yaw_degrees"), "views.val_yaw_degrees"),
        test_yaw_degrees=_float_tuple(_required(views, "test_yaw_degrees"), "views.test_yaw_degrees"),
        handedness_options=_handedness_tuple(_required(person_variation, "handedness"), "person_variation.handedness"),
        palm_width_range=_range_pair(_required(person_variation, "palm_width_range"), "person_variation.palm_width_range"),
        finger_length_scale_range=_range_pair(_required(person_variation, "finger_length_scale_range"), "person_variation.finger_length_scale_range"),
        speed_range_s=_range_pair(_required(motion, "speed_range_s"), "motion.speed_range_s"),
        start_pose=_start_pose_value(motion.get("start_pose", "fist"), "motion.start_pose"),
        start_hold_ratio=_fraction(_optional_float(motion, "start_hold_ratio", 0.20), "motion.start_hold_ratio"),
        transition_onset_range=_fraction_range_pair(
            motion.get("transition_onset_range", (0.18, 0.30)),
            "motion.transition_onset_range",
        ),
        finger_open_delay_range=_fraction_range_pair(
            motion.get("finger_open_delay_range", (0.0, 0.10)),
            "motion.finger_open_delay_range",
        ),
        motion_style=_motion_style_value(motion.get("motion_style", "staggered_finger_open"), "motion.motion_style"),
        abstain_before_motion=_bool_value(motion.get("abstain_before_motion", True), "motion.abstain_before_motion"),
        hesitation_probability=_probability(_required_float(motion, "hesitation_probability"), "motion.hesitation_probability"),
        noise_std=_non_negative_float(_required_float(motion, "noise_std"), "motion.noise_std"),
        family_ranges=family_ranges,
        extended_max=_probability(_required_float(ambiguity, "extended_max"), "ambiguity.extended_max"),
        flexed_min=_probability(_required_float(ambiguity, "flexed_min"), "ambiguity.flexed_min"),
    )


def validate_pose_family_config(config: PoseFamilyConfig) -> dict[str, object]:
    """Validate semantic ranges without writing a dataset."""

    findings: list[str] = []
    for label in config.labels:
        mid_curls = {finger: _midpoint(config.family_ranges[label][finger]) for finger in FINGER_NAMES}
        inferred = label_from_finger_curls(mid_curls, extended_max=config.extended_max, flexed_min=config.flexed_min)
        if inferred != label:
            findings.append(f"family {label} midpoint infers {inferred}")
    return {
        "status": "passed" if not findings else "failed",
        "findings": findings,
        "labels": list(config.labels),
        "keypoint_count": len(KEYPOINT_NAMES),
        "view_count": len(config.yaw_degrees) * len(config.pitch_degrees),
        "person_count": config.person_count,
    }


def sample_pose_family(
    config: PoseFamilyConfig,
    *,
    rng: np.random.Generator,
    label: PoseFamilyLabel,
    person_id: int,
    view_id: int,
) -> PoseFamilySample:
    """Sample one fist-start RPS or ambiguous trajectory."""

    if label not in config.labels:
        raise ValueError(f"label {label} is not enabled by the pose-family config")
    identity = person_identity(config, person_id)
    target_curls, target_spreads = _sample_semantic_ranges(config, rng=rng, label=label)
    yaw, pitch = view_pose(config, view_id)
    translation = cast(NDArray[np.float32], rng.normal(0.0, 0.03, size=(3,)).astype(np.float32))
    start_curls = _start_finger_curls(config.start_pose)
    start_spreads = _start_finger_spreads(config.start_pose)
    keypoints = np.zeros((config.sequence_length, len(KEYPOINT_NAMES), 3), dtype=np.float32)
    canonical = np.zeros_like(keypoints, dtype=np.float32)
    hesitation_center = float(rng.uniform(0.35, 0.65))
    hesitation_width = float(rng.uniform(0.06, 0.12))
    has_hesitation = bool(rng.random() < config.hesitation_probability)
    transition_onset = float(rng.uniform(config.transition_onset_range[0], config.transition_onset_range[1]))
    transition_onset = max(config.start_hold_ratio, transition_onset)
    finger_delays = {
        finger: float(rng.uniform(config.finger_open_delay_range[0], config.finger_open_delay_range[1]))
        if config.motion_style == "staggered_finger_open"
        else 0.0
        for finger in FINGER_NAMES
    }

    for frame_index in range(config.sequence_length):
        raw_progress = frame_index / max(1, config.sequence_length - 1)
        curls = {
            finger: start_curls[finger]
            + (target_curls[finger] - start_curls[finger])
            * _finger_motion_progress(
                raw_progress,
                onset=transition_onset + finger_delays[finger],
                has_hesitation=has_hesitation,
                hesitation_center=hesitation_center,
                hesitation_width=hesitation_width,
            )
            for finger in FINGER_NAMES
        }
        spreads = {
            finger: start_spreads[finger]
            + (target_spreads[finger] - start_spreads[finger])
            * _finger_motion_progress(
                raw_progress,
                onset=transition_onset + finger_delays[finger],
                has_hesitation=has_hesitation,
                hesitation_center=hesitation_center,
                hesitation_width=hesitation_width,
            )
            for finger in FINGER_NAMES
        }
        frame = build_keypoints_from_semantics(curls, spreads, identity)
        if identity.handedness == "left":
            frame = frame.copy()
            frame[:, 0] *= -1.0
        transformed = _apply_view_transform(frame, yaw_degrees=yaw, pitch_degrees=pitch) + translation
        if config.noise_std > 0.0 and raw_progress > config.start_hold_ratio:
            noise = cast(NDArray[np.float32], rng.normal(0.0, config.noise_std, size=transformed.shape).astype(np.float32))
            transformed = transformed + noise
        keypoints[frame_index] = transformed.astype(np.float32)
        canonical[frame_index] = canonicalize_keypoints(transformed)

    return PoseFamilySample(
        label=label,
        person_id=person_id,
        view_id=view_id,
        handedness=identity.handedness,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        keypoints=keypoints,
        canonical_keypoints=canonical,
        start_finger_curls=start_curls,
        start_finger_spreads=start_spreads,
        final_finger_curls=target_curls,
        final_finger_spreads=target_spreads,
    )


def generate_pose_family_dataset(config: PoseFamilyConfig) -> PoseFamilyDataset:
    """Generate a balanced identity-held-out pose-family dataset."""

    validation = validate_pose_family_config(config)
    if validation["status"] != "passed":
        raise ValueError(f"pose-family config failed validation: {validation['findings']}")
    rng = np.random.default_rng(config.seed)
    identity_split_by_person = build_identity_splits(
        person_count=config.person_count,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )
    view_count = len(config.yaw_degrees) * len(config.pitch_degrees)
    keypoints_rows: list[NDArray[np.float32]] = []
    canonical_rows: list[NDArray[np.float32]] = []
    start_curl_rows: list[list[float]] = []
    curl_rows: list[list[float]] = []
    labels: list[int] = []
    splits: list[int] = []
    identity_splits: list[int] = []
    view_splits: list[int] = []
    person_ids: list[int] = []
    view_ids: list[int] = []
    yaw_values: list[float] = []
    pitch_values: list[float] = []

    for person_id in range(config.person_count):
        for label_index, label in enumerate(config.labels):
            for sample_index in range(config.samples_per_label_per_person):
                view_id = int((person_id * len(config.labels) + label_index + sample_index) % view_count)
                sample = sample_pose_family(config, rng=rng, label=label, person_id=person_id, view_id=view_id)
                keypoints_rows.append(sample.keypoints)
                canonical_rows.append(sample.canonical_keypoints)
                start_curl_rows.append([sample.start_finger_curls[finger] for finger in FINGER_NAMES])
                curl_rows.append([sample.final_finger_curls[finger] for finger in FINGER_NAMES])
                labels.append(label_index)
                split_id = int(identity_split_by_person[person_id])
                splits.append(split_id)
                identity_splits.append(split_id)
                view_splits.append(view_split_id(config, sample.yaw_degrees))
                person_ids.append(person_id)
                view_ids.append(view_id)
                yaw_values.append(sample.yaw_degrees)
                pitch_values.append(sample.pitch_degrees)

    keypoint_array = cast(NDArray[np.float32], np.stack(keypoints_rows).astype(np.float32))
    canonical_array = cast(NDArray[np.float32], np.stack(canonical_rows).astype(np.float32))
    positions = cast(NDArray[np.float32], canonical_array.reshape(canonical_array.shape[0], canonical_array.shape[1], len(KEYPOINT_NAMES) * 3).astype(np.float32))
    velocities = _compute_frame_velocities(positions)
    sequences = cast(NDArray[np.float32], np.concatenate([positions, velocities], axis=2).astype(np.float32))
    return PoseFamilyDataset(
        sequences=sequences,
        positions=positions,
        velocities=velocities,
        keypoints=keypoint_array,
        canonical_keypoints=canonical_array,
        start_finger_curls=cast(NDArray[np.float32], np.asarray(start_curl_rows, dtype=np.float32)),
        final_finger_curls=cast(NDArray[np.float32], np.asarray(curl_rows, dtype=np.float32)),
        labels=cast(NDArray[np.int64], np.asarray(labels, dtype=np.int64)),
        splits=cast(NDArray[np.int64], np.asarray(splits, dtype=np.int64)),
        identity_splits=cast(NDArray[np.int64], np.asarray(identity_splits, dtype=np.int64)),
        view_splits=cast(NDArray[np.int64], np.asarray(view_splits, dtype=np.int64)),
        person_ids=cast(NDArray[np.int64], np.asarray(person_ids, dtype=np.int64)),
        view_ids=cast(NDArray[np.int64], np.asarray(view_ids, dtype=np.int64)),
        yaw_values=cast(NDArray[np.float32], np.asarray(yaw_values, dtype=np.float32)),
        pitch_values=cast(NDArray[np.float32], np.asarray(pitch_values, dtype=np.float32)),
        label_names=config.labels,
        split_names=SPLIT_NAMES,
        keypoint_names=KEYPOINT_NAMES,
    )


def save_pose_family_dataset(dataset: PoseFamilyDataset, config: PoseFamilyConfig) -> None:
    """Save pose-family tensors using the existing classifier-compatible schema."""

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_path,
        sequences=dataset.sequences,
        positions=dataset.positions,
        velocities=dataset.velocities,
        keypoints=dataset.keypoints,
        canonical_keypoints=dataset.canonical_keypoints,
        start_finger_curls=dataset.start_finger_curls,
        final_finger_curls=dataset.final_finger_curls,
        labels=dataset.labels,
        splits=dataset.splits,
        identity_splits=dataset.identity_splits,
        view_splits=dataset.view_splits,
        person_ids=dataset.person_ids,
        view_ids=dataset.view_ids,
        yaw_values=dataset.yaw_values,
        pitch_values=dataset.pitch_values,
        label_names=np.asarray(dataset.label_names),
        split_names=np.asarray(dataset.split_names),
        joint_names=np.asarray(dataset.keypoint_names),
        keypoint_names=np.asarray(dataset.keypoint_names),
    )
    metadata = {
        "config": _json_ready_config(config),
        "num_samples": int(dataset.labels.shape[0]),
        "sequence_length": int(dataset.sequences.shape[1]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "label_names": list(dataset.label_names),
        "split_names": list(dataset.split_names),
        "keypoint_names": list(dataset.keypoint_names),
        "start_pose": config.start_pose,
        "abstain_before_motion": config.abstain_before_motion,
        "identity_split_disjoint": _identity_split_disjoint(dataset),
        "view_split_note": "view_splits are stored separately from identity-held-out training splits",
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def audit_pose_family_dataset(dataset: PoseFamilyDataset) -> dict[str, object]:
    """Return label balance, split integrity, and semantic separability checks."""

    label_counts = _counts_by_name(dataset.labels, dataset.label_names)
    identity_split_counts = _counts_by_name(dataset.identity_splits, dataset.split_names)
    view_split_counts = _counts_by_name(dataset.view_splits, dataset.split_names)
    inferred = [
        label_from_finger_curls({finger: float(row[index]) for index, finger in enumerate(FINGER_NAMES)})
        for row in dataset.final_finger_curls
    ]
    inferred_ids = np.asarray([dataset.label_names.index(item) for item in inferred], dtype=np.int64)
    semantic_agreement = float(np.mean(inferred_ids == dataset.labels)) if dataset.labels.size else 0.0
    centroid_distances = _final_frame_centroid_distances(dataset)
    return {
        "status": "passed" if semantic_agreement >= 0.99 and _identity_split_disjoint(dataset) else "failed",
        "num_samples": int(dataset.labels.shape[0]),
        "feature_dim": int(dataset.sequences.shape[2]),
        "label_counts": label_counts,
        "identity_split_counts": identity_split_counts,
        "view_split_counts": view_split_counts,
        "identity_split_disjoint": _identity_split_disjoint(dataset),
        "semantic_label_agreement": semantic_agreement,
        "final_frame_centroid_distances": centroid_distances,
        "notes": [
            "Clear RPS classes are semantic pose families, not single fixed robot poses.",
            "Ambiguous examples remain labeled ambiguous for abstention instead of being forced into rock/paper/scissors.",
        ],
    }


def load_pose_family_dataset(path: Path) -> PoseFamilyDataset:
    """Load a saved pose-family dataset."""

    loaded = np.load(path, allow_pickle=False)
    return PoseFamilyDataset(
        sequences=cast(NDArray[np.float32], loaded["sequences"].astype(np.float32)),
        positions=cast(NDArray[np.float32], loaded["positions"].astype(np.float32)),
        velocities=cast(NDArray[np.float32], loaded["velocities"].astype(np.float32)),
        keypoints=cast(NDArray[np.float32], loaded["keypoints"].astype(np.float32)),
        canonical_keypoints=cast(NDArray[np.float32], loaded["canonical_keypoints"].astype(np.float32)),
        start_finger_curls=cast(
            NDArray[np.float32],
            loaded["start_finger_curls"].astype(np.float32)
            if "start_finger_curls" in loaded.files
            else np.zeros_like(loaded["final_finger_curls"], dtype=np.float32),
        ),
        final_finger_curls=cast(NDArray[np.float32], loaded["final_finger_curls"].astype(np.float32)),
        labels=cast(NDArray[np.int64], loaded["labels"].astype(np.int64)),
        splits=cast(NDArray[np.int64], loaded["splits"].astype(np.int64)),
        identity_splits=cast(NDArray[np.int64], loaded["identity_splits"].astype(np.int64)),
        view_splits=cast(NDArray[np.int64], loaded["view_splits"].astype(np.int64)),
        person_ids=cast(NDArray[np.int64], loaded["person_ids"].astype(np.int64)),
        view_ids=cast(NDArray[np.int64], loaded["view_ids"].astype(np.int64)),
        yaw_values=cast(NDArray[np.float32], loaded["yaw_values"].astype(np.float32)),
        pitch_values=cast(NDArray[np.float32], loaded["pitch_values"].astype(np.float32)),
        label_names=tuple(cast(Sequence[PoseFamilyLabel], loaded["label_names"].tolist())),
        split_names=tuple(cast(Sequence[str], loaded["split_names"].tolist())),
        keypoint_names=tuple(cast(Sequence[str], loaded["keypoint_names"].tolist())),
    )


def build_identity_splits(*, person_count: int, train_fraction: float, val_fraction: float) -> NDArray[np.int64]:
    """Assign each synthetic person to exactly one split."""

    if person_count < 3:
        raise ValueError("identity-held-out splitting requires at least 3 synthetic people")
    train_count = max(1, int(math.floor(person_count * train_fraction)))
    val_count = max(1, int(math.floor(person_count * val_fraction)))
    if train_count + val_count >= person_count:
        overflow = train_count + val_count - person_count + 1
        train_count = max(1, train_count - overflow)
    test_count = person_count - train_count - val_count
    if test_count <= 0:
        raise ValueError("identity-held-out split must leave at least one test person")
    return cast(NDArray[np.int64], np.asarray([0] * train_count + [1] * val_count + [2] * test_count, dtype=np.int64))


def person_identity(config: PoseFamilyConfig, person_id: int) -> PersonIdentity:
    """Return stable hand-shape parameters for a synthetic person."""

    rng = np.random.default_rng(config.seed + person_id * 1009)
    handedness = config.handedness_options[person_id % len(config.handedness_options)]
    finger_scales = {
        finger: float(rng.uniform(config.finger_length_scale_range[0], config.finger_length_scale_range[1]))
        for finger in FINGER_NAMES
    }
    return PersonIdentity(
        person_id=person_id,
        handedness=handedness,
        palm_width_m=float(rng.uniform(config.palm_width_range[0], config.palm_width_range[1])),
        finger_length_scales=finger_scales,
        thumb_angle_offset=float(rng.uniform(-0.18, 0.18)),
    )


def view_pose(config: PoseFamilyConfig, view_id: int) -> tuple[float, float]:
    """Return yaw/pitch for a configured view id."""

    grid = [(yaw, pitch) for yaw in config.yaw_degrees for pitch in config.pitch_degrees]
    if len(grid) == 0:
        raise ValueError("at least one configured camera view is required")
    return grid[view_id % len(grid)]


def view_split_id(config: PoseFamilyConfig, yaw_degrees: float) -> int:
    """Return split id for a camera yaw, independent from identity split."""

    if _contains_float(config.test_yaw_degrees, yaw_degrees):
        return 2
    if _contains_float(config.val_yaw_degrees, yaw_degrees):
        return 1
    return 0


def label_from_finger_curls(
    finger_curls: Mapping[str, float] | Mapping[FingerName, float],
    *,
    extended_max: float = 0.35,
    flexed_min: float = 0.65,
) -> PoseFamilyLabel:
    """Infer RPS semantic label from the non-thumb finger curl pattern."""

    values = {finger: float(finger_curls[finger]) for finger in CLASSIFIER_FINGERS}
    extended = {finger: values[finger] <= extended_max for finger in CLASSIFIER_FINGERS}
    flexed = {finger: values[finger] >= flexed_min for finger in CLASSIFIER_FINGERS}
    if all(flexed.values()):
        return "rock"
    if all(extended.values()):
        return "paper"
    if extended["index"] and extended["middle"] and flexed["ring"] and flexed["pinky"]:
        return "scissors"
    return "ambiguous"


def _start_finger_curls(start_pose: StartPose) -> dict[FingerName, float]:
    if start_pose == "fist":
        return {"thumb": 0.72, "index": 0.92, "middle": 0.92, "ring": 0.92, "pinky": 0.92}
    if start_pose == "neutral":
        return {finger: 0.18 for finger in FINGER_NAMES}
    raise ValueError(f"Unsupported start_pose: {start_pose}")


def _start_finger_spreads(start_pose: StartPose) -> dict[FingerName, float]:
    if start_pose in ("fist", "neutral"):
        return {finger: 0.0 for finger in FINGER_NAMES}
    raise ValueError(f"Unsupported start_pose: {start_pose}")


def _finger_motion_progress(
    raw_progress: float,
    *,
    onset: float,
    has_hesitation: bool,
    hesitation_center: float,
    hesitation_width: float,
) -> float:
    if raw_progress <= onset:
        return 0.0
    normalized = max(0.0, min(1.0, (raw_progress - onset) / max(1.0 - onset, 1e-6)))
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)
    if has_hesitation:
        hesitation = 0.12 * math.exp(-((raw_progress - hesitation_center) ** 2) / (2.0 * hesitation_width * hesitation_width))
        smooth = max(0.0, min(1.0, smooth - hesitation))
    return smooth


def build_keypoints_from_semantics(
    finger_curls: Mapping[FingerName, float],
    finger_spreads: Mapping[FingerName, float],
    identity: PersonIdentity,
) -> NDArray[np.float32]:
    """Build a right-hand 21-keypoint skeleton from semantic curl/spread features."""

    points = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32)
    width = identity.palm_width_m
    points[_KEYPOINT_INDEX["wrist"]] = np.asarray((0.0, 0.0, 0.0), dtype=np.float32)
    finger_specs: tuple[tuple[FingerName, str, float, float, tuple[float, float, float]], ...] = (
        ("index", "index", -0.30, 0.78, (0.45, 0.28, 0.20)),
        ("middle", "middle", -0.06, 0.84, (0.50, 0.32, 0.23)),
        ("ring", "ring", 0.18, 0.78, (0.46, 0.29, 0.21)),
        ("pinky", "pinky", 0.40, 0.66, (0.38, 0.24, 0.18)),
    )
    for finger, prefix, x_mul, y_mul, lengths in finger_specs:
        mcp = np.asarray((x_mul * width, y_mul * width, 0.0), dtype=np.float32)
        points[_KEYPOINT_INDEX[f"{prefix}_mcp"]] = mcp
        segment_lengths = (
            lengths[0] * width * identity.finger_length_scales[finger],
            lengths[1] * width * identity.finger_length_scales[finger],
            lengths[2] * width * identity.finger_length_scales[finger],
        )
        _write_finger_chain(
            points,
            prefix=prefix,
            start=mcp,
            segment_lengths=segment_lengths,
            curl=float(finger_curls[finger]),
            spread=float(finger_spreads[finger]) + x_mul * 0.10,
        )
    _write_thumb_chain(points, finger_curls=finger_curls, finger_spreads=finger_spreads, identity=identity)
    return points


def canonicalize_keypoints(keypoints: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize by wrist origin, palm plane, palm width, and handedness mirroring."""

    points = cast(NDArray[np.float32], np.asarray(keypoints, dtype=np.float32))
    if points.shape != (len(KEYPOINT_NAMES), 3):
        raise ValueError(f"keypoints must have shape {(len(KEYPOINT_NAMES), 3)}, got {points.shape}")
    origin = points[_KEYPOINT_INDEX["wrist"]]
    index_mcp = points[_KEYPOINT_INDEX["index_mcp"]]
    middle_mcp = points[_KEYPOINT_INDEX["middle_mcp"]]
    pinky_mcp = points[_KEYPOINT_INDEX["pinky_mcp"]]
    x_axis = _normalize(pinky_mcp - index_mcp)
    y_raw = _normalize(middle_mcp - origin)
    z_axis = _normalize(np.cross(x_axis, y_raw).astype(np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis).astype(np.float32))
    scale = float(np.linalg.norm(pinky_mcp - index_mcp))
    if scale <= 1e-8:
        raise ValueError("index_mcp and pinky_mcp must be separated for scale normalization")
    rel = (points - origin) / np.float32(scale)
    canonical = np.stack((rel @ x_axis, rel @ y_axis, rel @ z_axis), axis=1).astype(np.float32)
    if float(np.mean(canonical[list(TIP_INDICES), 2])) > 0.0:
        canonical[:, 2] *= -1.0
    return cast(NDArray[np.float32], canonical.astype(np.float32))


def _write_finger_chain(
    points: NDArray[np.float32],
    *,
    prefix: str,
    start: NDArray[np.float32],
    segment_lengths: tuple[float, float, float],
    curl: float,
    spread: float,
) -> None:
    current = start.copy()
    bend_multipliers = (0.95, 1.35, 1.70)
    for suffix, segment_length, bend_multiplier in zip(("pip", "dip", "tip"), segment_lengths, bend_multipliers, strict=True):
        bend = min(1.48, max(0.0, curl) * bend_multiplier)
        direction = np.asarray(
            (
                math.sin(spread) * math.cos(bend) * 0.55,
                math.cos(spread) * math.cos(bend),
                -math.sin(bend),
            ),
            dtype=np.float32,
        )
        current = current + direction * np.float32(segment_length)
        points[_KEYPOINT_INDEX[f"{prefix}_{suffix}"]] = current


def _write_thumb_chain(
    points: NDArray[np.float32],
    *,
    finger_curls: Mapping[FingerName, float],
    finger_spreads: Mapping[FingerName, float],
    identity: PersonIdentity,
) -> None:
    width = identity.palm_width_m
    curl = float(finger_curls["thumb"])
    spread = -0.86 - float(finger_spreads["thumb"]) * 0.45 + identity.thumb_angle_offset
    current = np.asarray((-0.54 * width, 0.28 * width, 0.0), dtype=np.float32)
    points[_KEYPOINT_INDEX["thumb_cmc"]] = current
    for suffix, length_mul, bend_mul in (("mcp", 0.34, 0.60), ("ip", 0.28, 0.95), ("tip", 0.24, 1.25)):
        bend = min(1.30, max(0.0, curl) * bend_mul)
        direction = np.asarray(
            (
                math.cos(spread) * math.cos(bend),
                math.sin(spread) * math.cos(bend) + 0.45 * curl,
                -math.sin(bend) * 0.75,
            ),
            dtype=np.float32,
        )
        current = current + direction * np.float32(length_mul * width * identity.finger_length_scales["thumb"])
        points[_KEYPOINT_INDEX[f"thumb_{suffix}"]] = current


def _sample_semantic_ranges(
    config: PoseFamilyConfig,
    *,
    rng: np.random.Generator,
    label: PoseFamilyLabel,
) -> tuple[dict[FingerName, float], dict[FingerName, float]]:
    ranges = config.family_ranges[label]
    curls = {finger: float(rng.uniform(ranges[finger][0], ranges[finger][1])) for finger in FINGER_NAMES}
    spread_low, spread_high = ranges["spread"]
    base_spread = float(rng.uniform(spread_low, spread_high))
    spreads: dict[FingerName, float] = {
        "thumb": base_spread * 1.4,
        "index": -base_spread,
        "middle": base_spread * 0.15,
        "ring": base_spread * 0.45,
        "pinky": base_spread,
    }
    return curls, spreads


def _apply_view_transform(keypoints: NDArray[np.float32], *, yaw_degrees: float, pitch_degrees: float) -> NDArray[np.float32]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rz = np.asarray(((math.cos(yaw), -math.sin(yaw), 0.0), (math.sin(yaw), math.cos(yaw), 0.0), (0.0, 0.0, 1.0)), dtype=np.float32)
    rx = np.asarray(((1.0, 0.0, 0.0), (0.0, math.cos(pitch), -math.sin(pitch)), (0.0, math.sin(pitch), math.cos(pitch))), dtype=np.float32)
    transform = rz @ rx
    return cast(NDArray[np.float32], (keypoints @ transform.T).astype(np.float32))


def _compute_frame_velocities(positions: NDArray[np.float32]) -> NDArray[np.float32]:
    velocities = np.zeros_like(positions, dtype=np.float32)
    velocities[:, 1:, :] = positions[:, 1:, :] - positions[:, :-1, :]
    velocities[:, 0, :] = velocities[:, 1, :] if positions.shape[1] > 1 else 0.0
    return velocities


def _final_frame_centroid_distances(dataset: PoseFamilyDataset) -> dict[str, float]:
    final_features = dataset.positions[:, -1, :]
    centroids: dict[str, NDArray[np.float32]] = {}
    for index, label in enumerate(dataset.label_names):
        if label == "ambiguous":
            continue
        label_mask = dataset.labels == index
        centroids[label] = cast(NDArray[np.float32], final_features[label_mask].mean(axis=0).astype(np.float32))
    distances: dict[str, float] = {}
    labels = sorted(centroids)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            distances[f"{left}_vs_{right}"] = float(np.linalg.norm(centroids[left] - centroids[right]))
    return distances


def _counts_by_name(values: NDArray[np.int64], names: Sequence[str]) -> dict[str, int]:
    return {name: int(np.sum(values == index)) for index, name in enumerate(names)}


def _identity_split_disjoint(dataset: PoseFamilyDataset) -> bool:
    person_sets = [set(dataset.person_ids[dataset.identity_splits == index].tolist()) for index in range(len(SPLIT_NAMES))]
    return person_sets[0].isdisjoint(person_sets[1]) and person_sets[0].isdisjoint(person_sets[2]) and person_sets[1].isdisjoint(person_sets[2])


def _json_ready_config(config: PoseFamilyConfig) -> dict[str, object]:
    raw = asdict(config)
    raw["output_path"] = str(config.output_path)
    raw["metadata_path"] = str(config.metadata_path)
    raw["audit_path"] = str(config.audit_path)
    return raw


def _normalize(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("cannot normalize a near-zero vector")
    return cast(NDArray[np.float32], (vector / np.float32(norm)).astype(np.float32))


def _contains_float(values: Sequence[float], target: float) -> bool:
    return any(math.isclose(value, target, abs_tol=1e-6) for value in values)


def _midpoint(pair: tuple[float, float]) -> float:
    return (pair[0] + pair[1]) / 2.0


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} must use string keys")
        parsed[key] = item
    return parsed


def _required(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    value = _required(mapping, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = _required(mapping, key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _optional_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _non_negative_float(value: float, label: str) -> float:
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _fraction(value: float, label: str) -> float:
    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{label} must be in (0, 1)")
    return value


def _probability(value: float, label: str) -> float:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return value


def _range_pair(value: object, label: str) -> tuple[float, float]:
    parsed = _ordered_range_pair(value, label)
    if parsed[0] < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _fraction_range_pair(value: object, label: str) -> tuple[float, float]:
    parsed = _ordered_range_pair(value, label)
    if parsed[0] < 0.0 or parsed[1] > 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return parsed


def _ordered_range_pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-item range")
    low, high = value
    if not isinstance(low, (int, float)) or isinstance(low, bool) or not isinstance(high, (int, float)) or isinstance(high, bool):
        raise ValueError(f"{label} must contain numbers")
    parsed = (float(low), float(high))
    if parsed[1] < parsed[0]:
        raise ValueError(f"{label} must be ordered")
    return parsed


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of numbers")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{label} must contain numbers")
        parsed.append(float(item))
    if len(parsed) == 0:
        raise ValueError(f"{label} must not be empty")
    return tuple(parsed)


def _bool_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _start_pose_value(value: object, label: str) -> StartPose:
    if value == "fist" or value == "neutral":
        return value
    raise ValueError(f"{label} must be 'fist' or 'neutral'")


def _motion_style_value(value: object, label: str) -> MotionStyle:
    if value == "smoothstep" or value == "staggered_finger_open":
        return value
    raise ValueError(f"{label} must be 'smoothstep' or 'staggered_finger_open'")


def _label_tuple(value: object, label: str) -> tuple[PoseFamilyLabel, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of labels")
    parsed: list[PoseFamilyLabel] = []
    for item in value:
        if not isinstance(item, str) or item not in POSE_FAMILY_LABELS:
            raise ValueError(f"{label} contains unsupported label {item!r}")
        parsed.append(item)
    if len(parsed) == 0:
        raise ValueError(f"{label} must not be empty")
    return tuple(parsed)


def _handedness_tuple(value: object, label: str) -> tuple[Handedness, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    parsed: list[Handedness] = []
    for item in value:
        if item not in ("right", "left"):
            raise ValueError(f"{label} contains unsupported handedness {item!r}")
        parsed.append(cast(Handedness, item))
    if len(parsed) == 0:
        raise ValueError(f"{label} must not be empty")
    return tuple(parsed)
