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

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving to file
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


def animate(args: argparse.Namespace) -> None:
    """
    Generate the animation described by *args*, which provides the
    ``input``, ``output``, ``fps``, ``interval`` and ``dpi`` attributes
    (as parsed by the ``animate-hand`` subcommand in ``digipy.cli``).
    """
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input file not found: {in_path}")

    frames = load_frames(in_path)
    if not frames:
        sys.exit("no frames loaded from input")
    print(f"loaded {len(frames)} frames")

    mins, maxs = compute_scene_bounds(frames)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Pre-extract per-frame data
    per_frame = [segments_for_frame(f) for f in frames]

    # Reusable artists
    skeleton_lines = []
    palm_scatter = None

    def init():
        nonlocal palm_scatter
        ax.clear()
        setup_axes(ax, mins, maxs)
        palm_scatter = ax.scatter([], [], [], c="red", s=50, depthshade=False)
        return []

    def setup_axes(ax, mins, maxs):
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        # Leap coordinates: x left/right, y up, z toward/away.
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("Hand tracking 3D animation")
        # Equal-ish aspect for 3D (matplotlib scales per-axis manually)
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            # Older matplotlib versions don't support set_box_aspect.
            pass

    def update(i):
        nonlocal skeleton_lines, palm_scatter
        # Remove old skeleton lines
        for ln in skeleton_lines:
            ln.remove()
        skeleton_lines = []

        segs, palms = per_frame[i]

        # Draw each bone segment
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
            skeleton_lines.append(ln)

        # Palm markers
        if palms:
            xs = [p[0] for p in palms]
            ys = [p[1] for p in palms]
            zs = [p[2] for p in palms]
            palm_scatter._offsets3d = (xs, ys, zs)

        # Slowly rotate the camera for a turntable view
        elev = 20 + 10 * np.sin(2 * np.pi * i / max(1, len(frames)))
        azim = (360.0 * i / max(1, len(frames))) % 360.0
        ax.view_init(elev=elev, azim=azim)

        ax.set_title(f"Hand tracking 3D animation  (frame {i + 1}/{len(frames)})")
        return skeleton_lines + [palm_scatter]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(per_frame),
        init_func=init,
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    out_path = Path(args.output)
    print(f"writing animation to {out_path} ...")
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer="pillow", fps=args.fps, dpi=args.dpi)
    else:
        # mp4 - try ffmpeg, fall back to a clearer error
        try:
            anim.save(str(out_path), writer="ffmpeg", fps=args.fps, dpi=args.dpi)
        except (ValueError, RuntimeError) as e:
            print(f"ffmpeg save failed ({e}); falling back to .gif")
            gif_out = out_path.with_suffix(".gif")
            anim.save(str(gif_out), writer="pillow", fps=args.fps, dpi=args.dpi)
            print(f"wrote {gif_out} instead")

    plt.close(fig)
    print("done")
