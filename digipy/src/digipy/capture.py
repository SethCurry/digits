import numpy as np
from pydantic import BaseModel


def parseBoneName(name: str):
    if name == "Metacarpal":
        return 1
    if name == "Proximal":
        return 2
    if name == "Middle":
        return 3
    if name == "Distal":
        return 4

    raise ValueError("unknown finger bone: " + name)


def parseFingerName(name: str):
    if name == "Thumb":
        return 1
    if name == "Index":
        return 2
    if name == "Middle":
        return 3
    if name == "Ring":
        return 4
    if name == "Pinky":
        return 5

    raise ValueError("unknown finger name: " + name)


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


def load_json_frames(path: str) -> list[Frame]:
    """Load JSONL frames from *path*, returning a list of frame dicts."""
    frames: list[Frame] = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            new_frame = Frame.model_validate_json(line)
            frames.append(new_frame)
    return frames


def to_training_data(frames: list[Frame]) -> list[np.array]:
    allMeasures: list[np.array] = []

    for frame in frames:
        for hand in frame.hands:
            measures: list[float] = []
            if hand.is_left:
                measures.append(0)
            else:
                measures.append(1)

            measures.append(hand.pitch)
            measures.append(hand.roll)
            measures.append(hand.yaw)
            # ensure all fingers present and in the same order
            fingers: dict[str, list[float]] = {
                "Thumb": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Index": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Middle": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Ring": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                "Pinky": [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            }
            for finger in hand.fingers:
                fingerMeasures: list[float] = []
                fingerMeasures.append(finger.length)
                fingerMeasures.append(finger.width)

                bones: dict[str, list[float]] = {
                    "Metacarpal": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                    "Proximal": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                    "Middle": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                    "Distal": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                }
                for bone in finger.bones:
                    boneMeasures: list[float] = []
                    boneMeasures.append(bone.start[0])
                    boneMeasures.append(bone.start[1])
                    boneMeasures.append(bone.start[2])
                    boneMeasures.append(bone.end[0])
                    boneMeasures.append(bone.end[1])
                    boneMeasures.append(bone.end[2])
                    boneMeasures.append(bone.direction[0])
                    boneMeasures.append(bone.direction[1])
                    boneMeasures.append(bone.direction[2])

                    bones[bone.name] = boneMeasures

                for b in ["Metacarpal", "Proximal", "Middle", "Distal"]:
                    fingerMeasures.extend(bones[b])
                fingers[finger.name] = fingerMeasures

            for fingerName in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                f = fingers[fingerName]

                measures.extend(f)

            allMeasures.append(np.array(measures, dtype=float))

        return allMeasures
