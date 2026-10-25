"""Gesture-recognition model and training pipeline.

Defines the convolutional gesture model used by ``digipy``, the
capture -> skeletal-frame conversion, and the dataset loading / training
helpers wired up by the ``train`` and ``inference`` subcommands of
:mod:`digipy.cli`.

Feature layout (66 channels per frame, 50 frames per sample):
  - 5 fingers x 4 knuckles x 3 (xyz) = 60
  - palm (3) + wrist (3) = 6
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow
from matplotlib import pyplot as plt
from pydantic import BaseModel
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import (
    AveragePooling1D,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical

from digipy.capture import Frame, load_json_frames
from digipy.gestures import parse_gesture

# Number of frames per sample and channels per frame.  Must match the model
# input shape and the constants in digipy.libdigits.
DURATION = 50
N_CHANNELS = 66


def _midpoint(one: list[float], two: list[float]) -> list[float]:
    """Return the midpoint of two 3-D coordinate vectors."""
    return [(one[0] + two[0]) / 2, (one[1] + two[1]) / 2, (one[2] + two[2]) / 2]


class Position(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_vector(cls, items: list[float]) -> Position:
        return cls(x=items[0], y=items[1], z=items[2])

    @classmethod
    def midpoint(cls, one: list[float], two: list[float]) -> Position:
        return cls.from_vector(_midpoint(one, two))

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class SkeletalFinger(BaseModel):
    first_knuckle: Position
    second_knuckle: Position
    third_knuckle: Position
    tip: Position


class SkeletalHGRFrame(BaseModel):
    """A single hand reduced to knuckle/tip/palm/wrist positions."""

    thumb: SkeletalFinger
    index: SkeletalFinger
    middle: SkeletalFinger
    ring: SkeletalFinger
    pinky: SkeletalFinger
    palm: Position
    wrist: Position

    @staticmethod
    def from_capture(frame: Frame) -> list[SkeletalHGRFrame]:
        """Convert a capture :class:`Frame` to one skeletal frame per hand."""
        skel_frames: list[SkeletalHGRFrame] = []

        for hand in frame.hands:
            wrist_position = Position.from_vector(hand.arm.wrist_position)
            palm_position = Position.from_vector(hand.palm_position)

            fingers: dict[str, SkeletalFinger] = {}
            for finger in hand.fingers:
                bones = {bone.name: bone for bone in finger.bones}

                tip = Position.from_vector(bones["Distal"].end)
                third_knuckle = Position.midpoint(
                    bones["Distal"].start, bones["Middle"].end
                )
                second_knuckle = Position.midpoint(
                    bones["Middle"].start,
                    bones["Proximal"].end,
                )
                first_knuckle = Position.midpoint(
                    bones["Proximal"].start,
                    bones["Metacarpal"].end,
                )

                fingers[finger.name] = SkeletalFinger(
                    first_knuckle=first_knuckle,
                    second_knuckle=second_knuckle,
                    third_knuckle=third_knuckle,
                    tip=tip,
                )

            skel_frames.append(
                SkeletalHGRFrame(
                    thumb=fingers["Thumb"],
                    index=fingers["Index"],
                    middle=fingers["Middle"],
                    ring=fingers["Ring"],
                    pinky=fingers["Pinky"],
                    palm=palm_position,
                    wrist=wrist_position,
                )
            )

        return skel_frames

    def to_frame(self) -> np.ndarray:
        """Flatten this skeletal frame into a 1-D feature vector of 66 floats."""
        dim: list[float] = []
        for finger in (self.thumb, self.index, self.middle, self.ring, self.pinky):
            for point in (
                finger.first_knuckle,
                finger.second_knuckle,
                finger.third_knuckle,
                finger.tip,
            ):
                dim.extend(point.to_list())
        dim.extend(self.palm.to_list())
        dim.extend(self.wrist.to_list())
        return np.array(dim, dtype=float)


@dataclass
class LoadedSample:
    answer: str
    samples: list[SkeletalHGRFrame]


@dataclass
class TrainingDatasets:
    """Train/validation feature and label arrays for model training.

    ``*_features`` have shape ``(n_samples, DURATION, N_CHANNELS)`` and
    ``*_labels`` are one-hot encoded with ``n_classes`` columns.
    """

    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray


def load_data_dir(
    path: str | Path, gestures: list[str], duration: int = DURATION
) -> list[LoadedSample]:
    """Load all labeled samples from a single sample directory.

    A sample directory contains an ``answer.txt`` with the gesture label and
    one or more ``*.json`` JSONL files, each a sequence of frames making up
    one or more windows.  Files whose frame count exceeds *duration* are
    transparently split into consecutive, non-overlapping windows of
    *duration* frames each; any trailing frames shorter than *duration*
    (including files that are shorter than *duration* to begin with) are
    discarded.  A capture frame may yield multiple skeletal frames (one per
    detected hand), so the windowing operates on skeletal frames rather than
    raw capture frames.
    *gestures* is the list of configured gesture names (class-index order)
    used to parse and validate the label.
    """
    path = Path(path)
    answer = parse_gesture(
        (path / "answer.txt").read_text(encoding="utf-8").strip(), gestures
    )

    all_samples: list[LoadedSample] = []
    for sample_file in sorted(f for f in os.listdir(path) if f.endswith(".json")):
        raw_frames = load_json_frames(path / sample_file)

        samples: list[SkeletalHGRFrame] = []
        for frame in raw_frames:
            samples.extend(SkeletalHGRFrame.from_capture(frame))

        for start in range(0, len(samples) - duration + 1, duration):
            all_samples.append(
                LoadedSample(answer=answer, samples=samples[start : start + duration])
            )

    return all_samples


def create_model(
    n_classes: int,
    duration: int = DURATION,
    n_channels: int = N_CHANNELS,
    dropout_probability: float = 0.2,
) -> Model:
    """Build the multi-branch per-channel convolutional gesture model.

    *n_classes* is the number of configured gestures (see
    ``digipy.cli.Config.gesture_names``)."""
    model_input = Input(shape=(duration, n_channels))

    # Slice into per-channel inputs so each channel gets its own conv branch.
    channel_inputs = Lambda(
        lambda x: tensorflow.split(x, num_or_size_splits=n_channels, axis=-1)
    )(model_input)

    features = []
    for channel in range(n_channels):
        channel_input = channel_inputs[channel]

        # High-kernel branch.
        high = Conv1D(filters=8, kernel_size=7, padding="same", activation="relu")(
            channel_input
        )
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding="same", activation="relu")(high)
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding="same", activation="relu")(high)
        high = Dropout(dropout_probability)(high)
        high = AveragePooling1D(pool_size=2)(high)

        # Low-kernel branch.
        low = Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(
            channel_input
        )
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding="same", activation="relu")(low)
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding="same", activation="relu")(low)
        low = Dropout(dropout_probability)(low)
        low = AveragePooling1D(pool_size=2)(low)

        # Pooling residual branch.
        ap_residual = AveragePooling1D(pool_size=2)(channel_input)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)

        features.append(concatenate([high, low, ap_residual]))

    features = concatenate(features)
    features = Flatten()(features)
    features = Dense(units=1936, activation="relu")(features)
    model_output = Dense(units=n_classes, activation="softmax")(features)

    return Model(inputs=[model_input], outputs=[model_output])


def load_sample_data(
    samples_dir: str | Path,
    gestures: list[str],
    training_per_validate: int = 4,
) -> TrainingDatasets:
    """Load all sample directories under *samples_dir* and split into
    train / validation datasets.

    *gestures* is the list of configured gesture names (class-index order);
    it defines the label space, so the number of classes is ``len(gestures)``.

    Every ``training_per_validate``-th sample (by load order) is held out for
    validation; the rest are used for training.
    """
    samples: list[LoadedSample] = []
    for entry in sorted(os.listdir(samples_dir)):
        samples.extend(load_data_dir(Path(samples_dir) / entry, gestures))

    train_split: list[LoadedSample] = []
    val_split: list[LoadedSample] = []
    for i, sample in enumerate(samples):
        if i % training_per_validate == 0:
            val_split.append(sample)
        else:
            train_split.append(sample)

    train_features = np.array(
        [np.stack([s.to_frame() for s in x.samples]) for x in train_split]
    )
    val_features = np.array(
        [np.stack([s.to_frame() for s in x.samples]) for x in val_split]
    )
    train_labels = to_categorical(
        np.array([gestures.index(x.answer) for x in train_split]),
        num_classes=len(gestures),
    )
    val_labels = to_categorical(
        np.array([gestures.index(x.answer) for x in val_split]),
        num_classes=len(gestures),
    )

    return TrainingDatasets(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
    )


def plot_history(history: History) -> None:
    """Plot training & validation accuracy and loss curves."""
    fig, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(10, 4))

    acc_ax.plot(history.history["accuracy"], label="train")
    acc_ax.plot(history.history["val_accuracy"], label="val")
    acc_ax.set_title("model accuracy")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.legend(loc="upper left")

    loss_ax.plot(history.history["loss"], label="train")
    loss_ax.plot(history.history["val_loss"], label="val")
    loss_ax.set_title("model loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


class GestureModel:
    """Wrapper around the Keras gesture model with train/save/predict helpers."""

    def __init__(self, gestures: list[str], load_model_path: str | Path | None = None):
        """Build a model with one output class per configured gesture.

        *gestures* is the list of configured gesture names (class-index
        order); index *i* of the model's softmax output corresponds to
        ``gestures[i]``.
        """
        self.gestures = gestures
        self.model = create_model(n_classes=len(gestures))
        if load_model_path:
            self.model.load_weights(load_model_path)

    def train(self, datasets: TrainingDatasets) -> History:
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return self.model.fit(
            datasets.train_features,
            datasets.train_labels,
            validation_data=(datasets.val_features, datasets.val_labels),
            epochs=100,
            batch_size=32,
        )

    def plot(self) -> None:
        plot_model(self.model, to_file="model.png", show_shapes=True)

    def save_weights(self, save_to: str | Path) -> None:
        self.model.save_weights(save_to)

    def categorize(self, frames: list[SkeletalHGRFrame]) -> list[int]:
        """Predict the gesture-class index for a single window of frames."""
        sample = np.stack([f.to_frame() for f in frames])[np.newaxis, ...]
        probabilities = self.model.predict(sample)
        return probabilities.argmax(axis=-1).tolist()

    def evaluate_accuracy(self, datasets: TrainingDatasets) -> tuple[float, float]:
        """Return ``(train_accuracy, val_accuracy)``."""
        _, train_accuracy = self.model.evaluate(
            datasets.train_features, datasets.train_labels
        )
        _, val_accuracy = self.model.evaluate(
            datasets.val_features, datasets.val_labels
        )
        return train_accuracy, val_accuracy
