"""
Generate a 3D animation from Leap-Motion-style hand tracking data in capture.json.

capture.json is JSONL: one JSON object per line. Each frame has:
    frame_id, hands: [{ arm: {elbow_position, wrist_position, direction},
                         fingers: [{ name, bones: [{name, start, end}, x4] }, x5],
                         palm_position, is_left, pitch, roll, yaw }]

This module draws the skeleton (arm + 5 fingers, 4 bones each) for every frame
and renders a rotating 3D animation, saving it to hand_animation.mp4 (or .gif).
It is exposed as the ``animate-hand`` subcommand of the ``digipy`` CLI; see
``digipy.cli``.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from digipy.hand_geometry import (  # noqa: F401 (re-exported for back-compat)
    BONE_ORDER,
    FINGER_ORDER,
    collect_segments,
    compute_scene_bounds,
    load_frames,
    segments_for_frame,
)


def _setup_axes(ax, mins, maxs) -> None:
    """Configure a 3D Axes with fixed bounds and Leap-coordinate labels."""
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # Leap coordinates: x left/right, y up, z toward/away.
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Hand tracking 3D animation")
    ax.set_box_aspect((1, 1, 1))


def _draw_skeleton(ax, segs):
    """Draw bone segments onto *ax*, returning the created line artists."""
    lines = []
    for s, e in segs:
        (ln,) = ax.plot(
            [s[0], e[0]],
            [s[1], e[1]],
            [s[2], e[2]],
            color="steelblue",
            linewidth=2.0,
            marker="o",
            markersize=3,
            markerfacecolor="steelblue",
        )
        lines.append(ln)
    return lines


def _save_animation(anim, out_path: Path, fps: int, dpi: int) -> None:
    """Save *anim* to *out_path*, falling back to .gif if ffmpeg is missing."""
    print(f"writing animation to {out_path} ...")
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
        return
    try:
        anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    except (ValueError, RuntimeError) as e:
        print(f"ffmpeg save failed ({e}); falling back to .gif")
        gif_out = out_path.with_suffix(".gif")
        anim.save(str(gif_out), writer="pillow", fps=fps, dpi=dpi)
        print(f"wrote {gif_out} instead")


def animate(args: argparse.Namespace) -> None:
    """Generate the animation described by *args*.

    *args* provides the ``input``, ``output``, ``fps``, ``interval`` and
    ``dpi`` attributes parsed by the ``animate-hand`` subcommand in
    :mod:`digipy.cli`.
    """
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input file not found: {in_path}")

    frames = load_frames(in_path)
    if not frames:
        sys.exit("no frames loaded from input")
    print(f"loaded {len(frames)} frames")

    mins, maxs = compute_scene_bounds(frames)
    per_frame = [segments_for_frame(f) for f in frames]
    n = len(per_frame)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    skeleton_lines: list = []
    palm_scatter = None

    def init():
        nonlocal palm_scatter
        ax.clear()
        _setup_axes(ax, mins, maxs)
        palm_scatter = ax.scatter([], [], [], c="red", s=50, depthshade=False)
        return []

    def update(i):
        nonlocal skeleton_lines, palm_scatter
        for ln in skeleton_lines:
            ln.remove()
        skeleton_lines = []

        segs, palms = per_frame[i]
        skeleton_lines = _draw_skeleton(ax, segs)

        if palms:
            palm_scatter._offsets3d = (
                [p[0] for p in palms],
                [p[1] for p in palms],
                [p[2] for p in palms],
            )

        # Slowly rotate the camera for a turntable view.
        elev = 20 + 10 * np.sin(2 * np.pi * i / max(1, n))
        azim = (360.0 * i / max(1, n)) % 360.0
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Hand tracking 3D animation  (frame {i + 1}/{n})")
        return skeleton_lines + [palm_scatter]

    anim = FuncAnimation(
        fig,
        update,
        frames=n,
        init_func=init,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    _save_animation(anim, Path(args.output), fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print("done")
