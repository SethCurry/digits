"""
Pure geometry helpers for Leap-Motion-style hand tracking captures.

Shared by ``animate_hand`` (offline animation rendering) and ``label_hand``
(interactive live viewer / training-sample exporter), so neither module has to
import the other (and therefore neither inherits the other's matplotlib
backend choice).

capture.json / sample.json is JSONL: one JSON object per line. Each frame has:
    frame_id, hands: [{ arm: {elbow_position, wrist_position, direction},
                         fingers: [{ name, bones: [{name, start, end}, x4] }, x5],
                         palm_position, is_left, pitch, roll, yaw }]
"""

import json
import sys

import numpy as np

FINGER_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
BONE_ORDER = ["Metacarpal", "Proximal", "Middle", "Distal"]


def load_frames(path):
    """Load JSONL frames from *path*, returning a list of frame dicts."""
    frames = []
    with open(path, "r") as f:
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


def collect_segments(hand):
    """
    Return a list of (start_xyz, end_xyz) line segments describing the skeleton
    for a single hand: the arm (elbow->wrist) plus every finger bone.
    """
    segs = []

    # Arm: elbow -> wrist
    arm = hand["arm"]
    elbow = np.array(arm["elbow_position"], dtype=float)
    wrist = np.array(arm["wrist_position"], dtype=float)
    segs.append((elbow, wrist))

    # Wrist -> each finger's metacarpal base, to suggest the palm
    fingers = {f["name"]: f for f in hand["fingers"]}
    for name in FINGER_ORDER:
        if name not in fingers:
            continue
        bones = {b["name"]: b for b in fingers[name]["bones"]}
        meta = bones.get("Metacarpal")
        if meta is not None:
            segs.append((wrist, np.array(meta["start"], dtype=float)))

    # Finger bones (metacarpal -> proximal -> middle -> distal)
    for name in FINGER_ORDER:
        if name not in fingers:
            continue
        bones = {b["name"]: b for b in fingers[name]["bones"]}
        for bname in BONE_ORDER:
            b = bones.get(bname)
            if b is None:
                continue
            start = np.array(b["start"], dtype=float)
            end = np.array(b["end"], dtype=float)
            segs.append((start, end))

    return segs


def segments_for_frame(frame):
    """Return (skeleton_segments, palm_points) for a frame dict."""
    all_segs = []
    palm_pts = []
    for hand in frame.get("hands", []):
        all_segs.extend(collect_segments(hand))
        palm_pts.append(np.array(hand["palm_position"], dtype=float))
    return all_segs, palm_pts


def compute_scene_bounds(frames, padding=30.0):
    """Compute axis limits that fit every frame, plus a little padding."""
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