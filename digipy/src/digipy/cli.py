#!/usr/bin/env python3

import argparse
import logging

import structlog
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


def _run_animate_hand(args: argparse.Namespace) -> None:
    # Imported lazily so that CLI startup (and other subcommands) do not pay
    # for the matplotlib/numpy imports of pywoop.animate_hand.
    from digipy.animate_hand import animate

    animate(args)


def _run_label_hand(args: argparse.Namespace) -> None:
    # Imported lazily: label_hand pulls in matplotlib's interactive widgets,
    # which we only want when actually launching the viewer.
    from digipy.label_hand import run

    run(args)


def _train_model(args: argparse.Namespace):
    model = training.GestureModel()
    model.plot()

    samples = training.load_sample_data(args.training_data)

    history = model.train(samples)

    training.plot_history(history)

    results = model.evaluate_accuracy(samples)

    train_accuracy = results[0]
    test_accuracy = results[1]

    print("Accuracy on training data was " + str(train_accuracy) + "%")
    print("Accuracy on test data was " + str(test_accuracy) + "%")


def build_parser() -> argparse.ArgumentParser:
    """Build the pywoop CLI parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="pywoop",
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
        default="./digits.model",
        help="The location to output the trained model to.",
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
    label.set_defaults(func=_run_label_hand)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
