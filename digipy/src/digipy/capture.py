"""Pydantic models for Leap-Motion-style hand tracking frames.

A capture.json / sample.json file is JSONL: one JSON object per line.  Each
line parses into a :class:`Frame`.  :func:`to_training_data` flattens the
skeletal data into the fixed-width feature vectors expected by the gesture
neural network.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel

BONE_ORDER = ["Metacarpal", "Proximal", "Middle", "Distal"]
FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Sentinel values used when a finger/bone is missing from a frame.  Kept at
# the same widths the original implementation used so feature-vector layout
# is unchanged.  In practice Leap Motion always reports all 5 fingers/4 bones
# so these placeholders are only a safety net.
_MISSING_FINGER = [-1.0] * 11
_MISSING_BONE = [-1.0] * 9  # start(3) + end(3) + direction(3)


def parse_bone_name(name: str) -> int:
    """Return the 1-based ordinal of a bone name."""
    try:
        return BONE_ORDER.index(name) + 1
    except ValueError:
        raise ValueError(f"unknown finger bone: {name}") from None


def parse_finger_name(name: str) -> int:
    """Return the 1-based ordinal of a finger name."""
    try:
        return FINGER_ORDER.index(name) + 1
    except ValueError:
        raise ValueError(f"unknown finger name: {name}") from None


class FrameBone(BaseModel):
    direction: list[float]
    end: list[float]
    name: str
    start: list[float]


class FrameFinger(BaseModel):
    bones: list[FrameBone]
    id: int
    length: float
    name: str
    width: float


class FrameArm(BaseModel):
    direction: list[float]
    elbow_position: list[float]
    wrist_position: list[float]


class FrameHand(BaseModel):
    arm: FrameArm
    id: int
    is_left: bool
    palm_position: list[float]
    fingers: list[FrameFinger]
    pitch: float
    roll: float
    yaw: float


class Frame(BaseModel):
    frame_id: int
    hands: list[FrameHand]


def load_json_frames(path: str | Path) -> list[Frame]:
    """Load JSONL frames from *path* into a list of :class:`Frame` objects."""
    frames: list[Frame] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(Frame.model_validate_json(line))
    return frames


def _flatten_bone(bone: FrameBone) -> list[float]:
    return [
        bone.start[0],
        bone.start[1],
        bone.start[2],
        bone.end[0],
        bone.end[1],
        bone.end[2],
        bone.direction[0],
        bone.direction[1],
        bone.direction[2],
    ]


def _flatten_finger(finger: FrameFinger) -> list[float]:
    """Flatten one finger into [length, width, <4 bones * 9 channels>]."""
    measures = [finger.length, finger.width]
    bones = {b.name: b for b in finger.bones}
    for bone_name in BONE_ORDER:
        bone = bones.get(bone_name)
        measures.extend(_flatten_bone(bone) if bone else list(_MISSING_BONE))
    return measures


def _flatten_hand(hand: FrameHand) -> list[float]:
    """Flatten one hand into the fixed-width feature vector."""
    measures: list[float] = [
        0.0 if hand.is_left else 1.0,
        hand.pitch,
        hand.roll,
        hand.yaw,
    ]

    fingers = {f.name: f for f in hand.fingers}
    for finger_name in FINGER_ORDER:
        finger = fingers.get(finger_name)
        measures.extend(_flatten_finger(finger) if finger else list(_MISSING_FINGER))

    return measures


def to_training_data(frames: list[Frame]) -> list[np.ndarray]:
    """Flatten every hand of every frame into a list of 1-D feature arrays."""
    all_measures: list[np.ndarray] = []
    for frame in frames:
        for hand in frame.hands:
            all_measures.append(np.array(_flatten_hand(hand), dtype=float))
    return all_measures
