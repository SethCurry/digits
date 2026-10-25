"""Pydantic models for Leap-Motion-style hand tracking frames.

A capture.json / sample.json file is JSONL: one JSON object per line.  Each
line parses into a :class:`Frame`.  The skeletal data is consumed by
:mod:`digipy.training`, which reduces each frame to the fixed-width feature
vectors expected by the gesture neural network.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(Frame.model_validate_json(line))
    return frames
