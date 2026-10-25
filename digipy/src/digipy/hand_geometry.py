"""Pure geometry helpers for Leap-Motion-style hand tracking captures.

Shared by :mod:`digipy.animate_hand` (offline animation rendering) and
:mod:`digipy.label_hand` (interactive live viewer / training-sample exporter),
so neither module has to import the other (and therefore neither inherits the
other's matplotlib backend choice).

capture.json / sample.json is JSONL: one JSON object per line.  Each frame has::

    frame_id, hands: [{ arm: {elbow_position, wrist_position, direction},
                         fingers: [{ name, bones: [{name, start, end}, x4] }, x5],
                         palm_position, is_left, pitch, roll, yaw }]
"""

import json
import sys
from pathlib import Path

import numpy as np

FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
BONE_ORDER = ["Metacarpal", "Proximal", "Middle", "Distal"]


def load_frames(path: str | Path) -> list[dict]:
    """Load JSONL frames from *path*, returning a list of frame dicts."""
    frames: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"warning: skipping invalid JSON on line {lineno}: {e}",
                    file=sys.stderr,
                )
    return frames


def collect_segments(hand: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(start_xyz, end_xyz)`` line segments for a single hand.

    Covers the arm (elbow -> wrist) plus every finger bone, with extra
    wrist -> metacarpal-base segments to suggest the palm.
    """
    segs: list[tuple[np.ndarray, np.ndarray]] = []

    arm = hand["arm"]
    elbow = np.array(arm["elbow_position"], dtype=float)
    wrist = np.array(arm["wrist_position"], dtype=float)
    segs.append((elbow, wrist))

    # Look up each finger's bones once, reused for both the palm and bone
    # segments below.
    finger_bones: dict[str, dict[str, dict]] = {}
    for finger in hand["fingers"]:
        finger_bones[finger["name"]] = {b["name"]: b for b in finger["bones"]}

    # Wrist -> each finger's metacarpal base, to suggest the palm.
    for name in FINGER_ORDER:
        bones = finger_bones.get(name)
        if bones and (meta := bones.get("Metacarpal")):
            segs.append((wrist, np.array(meta["start"], dtype=float)))

    # Finger bones: metacarpal -> proximal -> middle -> distal.
    for name in FINGER_ORDER:
        bones = finger_bones.get(name)
        if not bones:
            continue
        for bname in BONE_ORDER:
            b = bones.get(bname)
            if b is None:
                continue
            segs.append(
                (np.array(b["start"], dtype=float), np.array(b["end"], dtype=float))
            )

    return segs


def segments_for_frame(
    frame: dict,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Return ``(skeleton_segments, palm_points)`` for a frame dict."""
    all_segs: list[tuple[np.ndarray, np.ndarray]] = []
    palm_pts: list[np.ndarray] = []
    for hand in frame.get("hands", []):
        all_segs.extend(collect_segments(hand))
        palm_pts.append(np.array(hand["palm_position"], dtype=float))
    return all_segs, palm_pts


def compute_scene_bounds(
    frames: list[dict], padding: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute axis limits (mins, maxs) that fit every frame, plus padding."""
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for frame in frames:
        segs, palms = segments_for_frame(frame)
        for s, e in segs:
            mins = np.minimum(mins, s)
            maxs = np.maximum(maxs, e)
        for p in palms:
            mins = np.minimum(mins, p)
            maxs = np.maximum(maxs, p)
    mins -= padding
    maxs += padding
    return mins, maxs
