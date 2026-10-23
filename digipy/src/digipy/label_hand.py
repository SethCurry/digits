"""
Live, interactive 3D viewer for Leap-Motion-style hand tracking captures.

This is the interactive counterpart to ``animate_hand``: instead of rendering a
fixed rotating animation to a file, it opens a matplotlib window showing one
frame at a time of a capture.json (JSONL) and gives you buttons to scrub
through the capture and export windows of consecutive frames as labeled
training samples for the gesture neural network.

Each exported sample is written as a directory under the output dataset root
containing::

    <output_root>/<sample_name>/answer.txt   # gesture label, e.g. "conductor_up"
    <output_root>/<sample_name>/sample.json  # JSONL, one frame per line

which matches the layout consumed by ``digipy.training.load_sample_data``.

Exposed as the ``label-hand`` subcommand of the ``digipy`` CLI; see
``digipy.cli``.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

from digipy.gestures import Gesture, parse_gesture
from digipy.hand_geometry import (
    compute_scene_bounds,
    load_frames,
    segments_for_frame,
)

# Known gesture labels, used as defaults/suggestions in the UI.
GESTURE_LABELS = [g.name.lower() for g in Gesture]


def _draw_frame(ax, segs, palms):
    """Draw skeleton segments + palm markers onto a 3D Axes, returning artists."""
    artists = []
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
        artists.append(ln)
    if palms:
        xs = [p[0] for p in palms]
        ys = [p[1] for p in palms]
        zs = [p[2] for p in palms]
        sc = ax.scatter(xs, ys, zs, c="red", s=50, depthshade=False)
        artists.append(sc)
    return artists


def _export_sample(frames, start, count, label, sample_name, out_root):
    """
    Write frames[start : start+count] to ``out_root/<sample_name>`` as a labeled
    training sample (answer.txt + sample.json). Returns the path written.
    Raises ValueError if the inputs are invalid.
    """
    label = (label or "").strip()
    if not label:
        raise ValueError("no gesture label set")
    # Validate the label parses to a known gesture.
    parse_gesture(label)

    count = int(count)
    if count <= 0:
        raise ValueError("frame count must be a positive integer")

    start = int(start)
    if start < 0 or start + count > len(frames):
        raise ValueError(
            f"window [{start}, {start + count}) is out of range "
            f"(capture has {len(frames)} frames)"
        )

    sample_name = (sample_name or "").strip()
    if not sample_name:
        sample_name = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    out_dir = Path(out_root) / sample_name
    if out_dir.exists():
        raise ValueError(f"sample directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    (out_dir / "answer.txt").write_text(label + "\n", encoding="utf-8")

    window = frames[start : start + count]
    (out_dir / "sample.json").write_text(
        "".join(json.dumps(frame) + "\n" for frame in window),
        encoding="utf-8",
    )

    return out_dir


def run(args: argparse.Namespace) -> None:
    """
    Open the interactive viewer described by *args* (``input``,
    ``output``, ``play_interval``), as parsed by the ``label-hand``
    subcommand in ``digipy.cli``.
    """
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"input file not found: {in_path}")

    frames = load_frames(in_path)
    if not frames:
        sys.exit("no frames loaded from input")
    print(f"loaded {len(frames)} frames from {in_path}")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"samples will be written under {out_root}")

    mins, maxs = compute_scene_bounds(frames)
    n = len(frames)

    # --- Layout: 3D scene on top, control strip below ---------------------
    fig = plt.figure(figsize=(9, 9))
    fig.canvas.manager.set_window_title("digipy label-hand")

    # 3D axes for the hand skeleton
    ax = fig.add_axes([0.05, 0.30, 0.90, 0.65], projection="3d")
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        # Older matplotlib versions don't support set_box_aspect.
        pass

    status_ax = fig.add_axes([0.05, 0.255, 0.90, 0.03])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.0, 0.5, "", va="center", ha="left", fontsize=10, family="monospace"
    )

    current = {"i": 0}
    skeleton_artists = []

    def render(i):
        # Clamp to valid range.
        i = max(0, min(n - 1, i))
        current["i"] = i
        for art in skeleton_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                # Artist may already be detached from the axes.
                pass
        skeleton_artists.clear()
        segs, palms = segments_for_frame(frames[i])
        skeleton_artists.extend(_draw_frame(ax, segs, palms))
        ax.set_title(f"Hand tracking 3D  (frame {i + 1}/{n})")
        status_text.set_text(_status(i))
        if frame_slider.val != i:
            frame_slider.eventson = False
            frame_slider.set_val(i)
            frame_slider.eventson = True
        fig.canvas.draw_idle()

    def _status(i):
        remaining = n - i
        return (
            f"frame {i + 1}/{n}   "
            f"remaining={remaining}   "
            f'label="{label_box.text}"   count={count_box.text}   '
            f"out={out_root}"
        )

    # --- Controls ---------------------------------------------------------
    # [ Prev ] [ Play/Pause ] [ Next ]   on one row
    prev_ax = fig.add_axes([0.05, 0.20, 0.12, 0.05])
    play_ax = fig.add_axes([0.20, 0.20, 0.14, 0.05])
    next_ax = fig.add_axes([0.37, 0.20, 0.12, 0.05])
    btn_prev = Button(prev_ax, "<< Prev")
    btn_play = Button(play_ax, "Play")
    btn_next = Button(next_ax, "Next >>")

    # Frame slider
    slider_ax = fig.add_axes([0.55, 0.205, 0.40, 0.04])
    frame_slider = Slider(
        slider_ax, "frame", 0, max(1, n - 1), valinit=0, valstep=1, valfmt="%0.0f"
    )

    # Label text box
    label_ax = fig.add_axes([0.05, 0.13, 0.30, 0.05])
    label_box = TextBox(label_ax, "label", initial=GESTURE_LABELS[0])

    # Count text box (how many frames to save)
    count_ax = fig.add_axes([0.40, 0.13, 0.14, 0.05])
    count_box = TextBox(count_ax, "save count", initial="20")

    # Sample name (optional); blank => auto-generated
    name_ax = fig.add_axes([0.05, 0.06, 0.49, 0.05])
    name_box = TextBox(name_ax, "sample name (blank=auto)", initial="")

    # Save button
    save_ax = fig.add_axes([0.60, 0.06, 0.35, 0.05])
    btn_save = Button(save_ax, "Save next N frames")

    # --- Callbacks --------------------------------------------------------
    def on_prev(_event):
        render(current["i"] - 1)

    def on_next(_event):
        render(current["i"] + 1)

    def on_slider(val):
        render(int(val))

    playing = {"on": False}

    def on_play(_event):
        if playing["on"]:
            playing["on"] = False
            btn_play.label.set_text("Play")
            return
        playing["on"] = True
        btn_play.label.set_text("Pause")

        # Blocking-ish playback loop; plt.pause pumps GUI events and sleeps.
        while playing["on"] and plt.fignum_exists(fig.number):
            nxt = current["i"] + 1
            if nxt >= n:
                nxt = 0  # loop
            render(nxt)
            plt.pause(args.play_interval / 1000.0)
        if plt.fignum_exists(fig.number):
            btn_play.label.set_text("Play")
            fig.canvas.draw_idle()

    def on_save(_event):
        try:
            out_dir = _export_sample(
                frames,
                start=current["i"],
                count=count_box.text,
                label=label_box.text,
                sample_name=name_box.text,
                out_root=out_root,
            )
            msg = f"saved {count_box.text} frames -> {out_dir}"
            print(msg)
            status_text.set_text(msg)
        except ValueError as e:
            status_text.set_text(f"ERROR: {e}")
            print(f"save failed: {e}", file=sys.stderr)
        fig.canvas.draw_idle()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_play.on_clicked(on_play)
    btn_save.on_clicked(on_save)
    frame_slider.on_changed(on_slider)

    # Keyboard shortcuts: left/right to scrub, space to play/pause,
    # 's' to save.
    def on_key(event):
        if event.key == "left":
            on_prev(None)
        elif event.key == "right":
            on_next(None)
        elif event.key == " ":
            on_play(None)
        elif event.key == "s":
            on_save(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial draw
    render(0)
    print(
        "controls: Prev/Next buttons or <-/-> arrows, space = play/pause, "
        "s = save next N frames"
    )
    plt.show()
