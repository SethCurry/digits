"""Command-line entry point for the ``digipy`` toolkit.

Subcommands:
  train         Train a gesture model from labeled samples.
  animate-hand  Render a 3D animation from a capture.json file.
  label-hand    Interactive 3D viewer / training-sample exporter.
  inference     Run live gesture inference from the digits binary.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
from pathlib import Path

import structlog
import yaml
from pydantic import BaseModel

from digipy import training

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
log = structlog.get_logger()


class ConfigGesture(BaseModel):
    training_data_dir: str
    on_trigger: str | None = None
    on_end: str | None = None
    debounce_seconds: int | None = None


class Config(BaseModel):
    gestures: dict[str, ConfigGesture]
    interval_ms: int = 20
    duration_frames: int = 50
    debounce_seconds: int = 1
    on_trigger: str | None = None
    on_end: str | None = None

    @property
    def gesture_names(self) -> list[str]:
        """The configured gesture names, in class-index order.

        The position of a gesture in this list is the index of its class in
        the model's output, so the order in the config file must stay stable
        between training and inference.
        """
        return sorted(self.gestures)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config` instance.

    See ``docs/config.sample.yml`` for the expected format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"config file not found: {path} "
            "(copy docs/config.sample.yml to get started)"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)


def _run_animate_hand(args: argparse.Namespace) -> None:
    # Imported lazily so that CLI startup (and other subcommands) do not pay
    # for the matplotlib/numpy imports of digipy.animate_hand.
    from digipy.animate_hand import animate

    animate(args)


def _run_label_hand(args: argparse.Namespace) -> None:
    # Imported lazily: label_hand pulls in matplotlib's interactive widgets,
    # which we only want when actually launching the viewer.
    from digipy.label_hand import run

    gestures = load_config(args.config).gesture_names
    run(args, gestures)


def _train_model(args: argparse.Namespace) -> None:
    """Train a gesture model from labeled samples and save the weights."""
    # The set of gestures (and thus the model's output size) comes from the
    # config file rather than a hard-coded list.
    gestures = load_config(args.config).gesture_names
    model = training.GestureModel(gestures)

    samples = training.load_sample_data(args.training_data, gestures)
    history = model.train(samples)
    training.plot_history(history)
    model.save_weights(args.output)


def spawn_command(cmd: str):
    threading.Thread(
        target=subprocess.run, args=(cmd), kwargs={"shell": True}, daemon=True
    ).start()


def _run_inference(args: argparse.Namespace) -> None:
    # Imported lazily so CLI startup for other subcommands doesn't pay for the
    # tensorflow import.
    import numpy as np

    from digipy import libdigits
    from digipy.training import GestureModel

    # Gesture names (in class-index order) come from the config file.
    config = load_config(args.config)
    gestures = config.gestures
    gesture_names = config.gesture_names

    libdigits.start_listener(args.interval, args.duration)

    log.info("loading model weights", model_path=args.model)
    model = GestureModel(gesture_names, load_model_path=args.model)
    log.info("model loaded")

    last_gesture: str | None = None

    try:
        while True:
            sample = libdigits.get_frames()
            # Model input is (None, duration, n_channels); add the batch dim.
            sample = sample[np.newaxis, ...]
            probabilities = model.model.predict(sample, verbose=1)
            predicted_class = int(probabilities.argmax(axis=-1)[0])

            gesture_name = gesture_names[predicted_class]
            log.info(
                "gesture detected",
                gesture=gesture_name,
                confidence=float(probabilities[0][predicted_class]),
            )

            if gesture_name in gestures and gesture_name != last_gesture:
                gesture = gestures["gesture_name"]

                if gesture.on_trigger != None:
                    spawn_command(gesture.on_trigger)

                if last_gesture is not None and last_gesture in gestures:
                    found_last = gestures[last_gesture]

                    if found_last.on_end is not None:
                        spawn_command(found_last.on_end)

            last_gesture = gesture_name

    except KeyboardInterrupt:
        log.info("interrupted")


def build_parser() -> argparse.ArgumentParser:
    """Build the digipy CLI parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="digipy",
        description="Tools for Leap-Motion-style hand tracking data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    train = subparsers.add_parser(
        "train", help="Train a model", description="Train a model."
    )
    train.add_argument(
        "-t",
        "--training-data",
        default="../training/data",
        help="The directory to load training data from.",
    )
    train.add_argument(
        "-o",
        "--output",
        default="./model.weights.h5",
        help="The location to output the trained model weights to.",
    )
    train.add_argument(
        "-c",
        "--config",
        default="../config.yml",
        help=(
            "Path to the YAML config file defining the gesture set "
            "(see docs/config.sample.yml)."
        ),
    )
    train.set_defaults(func=_train_model)

    animate = subparsers.add_parser(
        "animate-hand",
        help="generate a 3D animation from hand tracking data",
        description=(
            "Generate a 3D animation from Leap-Motion-style hand tracking "
            "data in capture.json. The input is JSONL: one JSON object per "
            "line, each with a frame_id and hands (arm + 5 fingers, 4 bones "
            "each). Draws the skeleton of every frame and renders a rotating "
            "3D animation, saving it to an .mp4 (or .gif) file."
        ),
    )
    animate.add_argument(
        "input", nargs="?", default="capture.json", help="path to capture.json (JSONL)"
    )
    animate.add_argument(
        "-o",
        "--output",
        default="hand_animation.mp4",
        help="output file (.mp4 or .gif)",
    )
    animate.add_argument("--fps", type=int, default=30, help="frames per second")
    animate.add_argument(
        "--interval", type=int, default=33, help="ms between frames (used for gif)"
    )
    animate.add_argument("--dpi", type=int, default=120)
    animate.set_defaults(func=_run_animate_hand)

    label = subparsers.add_parser(
        "label-hand",
        help="interactive 3D viewer / training-sample exporter",
        description=(
            "Open a live 3D viewer for a capture.json (JSONL) and scrub "
            "through frames one at a time. Export windows of consecutive "
            "frames as labeled training samples (answer.txt + sample.json) "
            "for the gesture neural network. Each saved sample is written "
            "as a directory under --output (default: ../training/data)."
        ),
    )
    label.add_argument(
        "input",
        nargs="?",
        default="capture.json",
        help="path to capture.json (JSONL)",
    )
    label.add_argument(
        "-o",
        "--output",
        default="../training/data",
        help="dataset root directory to write new labeled samples into",
    )
    label.add_argument(
        "--play-interval",
        type=int,
        default=33,
        help="ms between frames when playing (default 33 ~= 30 fps)",
    )
    label.add_argument(
        "-c",
        "--config",
        default="../config.yml",
        help=(
            "Path to the YAML config file defining the gesture set "
            "(see docs/config.sample.yml)."
        ),
    )
    label.set_defaults(func=_run_label_hand)

    inference = subparsers.add_parser(
        "inference",
        help="run live gesture inference",
        description=(
            "Load the trained gesture model and start the digits binary "
            "to stream live hand-tracking frames. A sliding window of "
            "frames is fed through the model and the detected gesture is "
            "printed for each window."
        ),
    )
    inference.add_argument(
        "-m",
        "--model",
        default="./model.weights.h5",
        help="Path to the trained model weights (.weights.h5).",
    )
    inference.add_argument(
        "-b",
        "--binary",
        default="../src/build/digits",
        help=(
            "Path to the digits binary (currently informational only; the "
            "C listener connects to the Leap service directly)."
        ),
    )
    inference.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Frame emit interval (ms) passed to the digits binary.",
    )
    inference.add_argument(
        "--duration",
        type=int,
        default=50,
        help="Number of frames per inference window (model input length).",
    )
    inference.add_argument(
        "-c",
        "--config",
        default="../config.yml",
        help=(
            "Path to the YAML config file defining the gesture set "
            "(see docs/config.sample.yml)."
        ),
    )
    inference.set_defaults(func=_run_inference)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
