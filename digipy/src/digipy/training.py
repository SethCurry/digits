import os
from dataclasses import dataclass

import numpy
import tensorflow
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from pydantic import BaseModel
from tensorflow.keras.backend import expand_dims
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
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical

from digipy.capture import Frame, load_json_frames
from digipy.gestures import Gesture, parse_gesture


def average(one: float, two: float):
    return (one + two) / 2


def average_vectors(one: list[float], two: list[float]):
    return [average(one[0], two[0]), average(one[1], two[1]), average(one[2], two[2])]


class Position(BaseModel):
    x: float
    y: float
    z: float

    @staticmethod
    def from_vector(items: list[float]):
        return Position(x=items[0], y=items[1], z=items[2])

    @staticmethod
    def avg_vectors(one: list[float], two: list[float]):
        return Position.from_vector(average_vectors(one, two))

    def to_list(self):
        return [self.x, self.y, self.z]


class SkeletalFinger(BaseModel):
    first_knuckle: Position
    second_knuckle: Position
    third_knuckle: Position
    tip: Position


class SkeletalHGRFrame(BaseModel):
    thumb: SkeletalFinger
    index: SkeletalFinger
    middle: SkeletalFinger
    ring: SkeletalFinger
    pinky: SkeletalFinger
    palm: Position
    wrist: Position

    @staticmethod
    def from_capture(frame: Frame) -> list["SkeletalHGRFrame"]:
        skelFrames = []

        for hand in frame.hands:
            wrist_position = Position.from_vector(hand.arm.wrist_position)
            palm_position = Position.from_vector(hand.palm_position)

            fingers = {}
            for finger in hand.fingers:
                bones = {}
                for bone in finger.bones:
                    bones[bone.name] = bone

                tip = Position.from_vector(bones["Distal"].end)
                third_knuckle = Position.avg_vectors(
                    bones["Distal"].start, bones["Middle"].end
                )
                second_knuckle = Position.avg_vectors(
                    bones["Middle"].start,
                    bones["Proximal"].end,
                )
                first_knuckle = Position.avg_vectors(
                    bones["Proximal"].start,
                    bones["Metacarpal"].end,
                )

                fingers[finger.name] = SkeletalFinger(
                    first_knuckle=first_knuckle,
                    second_knuckle=second_knuckle,
                    third_knuckle=third_knuckle,
                    tip=tip,
                )

            skelFrame = SkeletalHGRFrame(
                thumb=fingers["Thumb"],
                index=fingers["Index"],
                middle=fingers["Middle"],
                ring=fingers["Ring"],
                pinky=fingers["Pinky"],
                palm=palm_position,
                wrist=wrist_position,
            )

            skelFrames.append(skelFrame)

        return skelFrames

    def to_frame(self):
        dim: list[float] = []

        for i in [self.thumb, self.index, self.middle, self.ring, self.pinky]:
            for j in [i.first_knuckle, i.second_knuckle, i.third_knuckle, i.tip]:
                dim.extend(j.to_list())

        dim.extend(self.palm.to_list())
        dim.extend(self.wrist.to_list())
        return numpy.array(dim, dtype=float)


@dataclass
class LoadedSample:
    answer: Gesture
    samples: list[SkeletalHGRFrame]


def load_data_dir(path: str, duration: int = 50):
    answer_path = path + "/answer.txt"

    all_samples = []

    sample_files = [x for x in os.listdir(path) if x.endswith(".json")]

    with open(answer_path, "r") as fd:
        answer_text = fd.read().strip()

    answer = parse_gesture(answer_text)

    for sample_file in sample_files:
        sample_path = path + "/" + sample_file
        raw_samples = load_json_frames(sample_path)

        samples = []
        [samples.extend(SkeletalHGRFrame.from_capture(x)) for x in raw_samples]

        if len(samples) != duration:
            continue

        loaded = LoadedSample(answer=answer, samples=samples)
        all_samples.append(loaded)

    return all_samples


force_use_cpu = False

if force_use_cpu:
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from tensorflow.keras import backend as K

    print(
        "There should not be a GPU in the following list:",
        K.tensorflow_backend._get_available_gpus(),
    )


def create_model(n_classes=3, duration=50, n_channels=66, dropout_probability=0.2):
    # Define model, using functional API
    model_input = Input(shape=(duration, n_channels))

    # slice into channels
    channel_inputs = Lambda(
        lambda x: tensorflow.split(x, num_or_size_splits=n_channels, axis=-1)
    )(model_input)

    features = []
    for channel in range(n_channels):
        channel_input = channel_inputs[channel]
        # high branch
        high = Conv1D(
            filters=8,
            kernel_size=7,
            padding="same",
            activation="relu",
            input_shape=(100, 1),
        )(channel_input)
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding="same", activation="relu")(high)
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding="same", activation="relu")(high)
        high = Dropout(dropout_probability)(high)
        high = AveragePooling1D(pool_size=2)(high)
        # low branch
        low = Conv1D(
            filters=8,
            kernel_size=3,
            padding="same",
            activation="relu",
            input_shape=(100, 1),
        )(channel_input)
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding="same", activation="relu")(low)
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding="same", activation="relu")(low)
        low = Dropout(dropout_probability)(low)
        low = AveragePooling1D(pool_size=2)(low)
        # pooling branch
        ap_residual = AveragePooling1D(pool_size=2, input_shape=(100, 1))(channel_input)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)
        # channel output
        channel_output = concatenate([high, low, ap_residual])
        features.append(channel_output)

    features = concatenate(features)
    features = Flatten()(features)
    features = Dense(units=1936, activation="relu")(features)

    model_output = Dense(units=n_classes, activation="softmax")(features)

    model = Model(inputs=[model_input], outputs=[model_output])
    return model


# We load a gesture dataset:
#
#   x.shape should be (dataset_size, duration, channel)
#   y.shape should be (dataset_size, 1)


@dataclass
class TrainingDatasets:
    x_train: numpy.array
    x_test: numpy.array
    y_train: numpy.array
    y_test: numpy.array


def load_sample_data(
    samples_dir: str = "../training/data", training_per_validate: int = 4, n_classes: int = 3
):
    sample_dirs = os.listdir(samples_dir)
    samples = []
    [samples.extend(load_data_dir(samples_dir + "/" + x)) for x in sample_dirs]

    x_samples: list[LoadedSample] = []
    y_samples: list[LoadedSample] = []

    for i, t in enumerate(samples):
        if len(t.samples) != 50:
            continue
        if i % training_per_validate == 0:
            y_samples.append(t)
        else:
            x_samples.append(t)


    x_train = numpy.array([list(map(lambda i: i.to_frame(), x.samples)) for x in x_samples])
    x_test = numpy.array([x.answer.value for x in x_samples])
    y_train = numpy.array([list(map(lambda i: i.to_frame(), x.samples)) for x in y_samples])
    y_test = numpy.array([x.answer.value for x in y_samples])

    # Transform the labels to one-hot encoding for the cross-entropy loss
    y_test = to_categorical(y_test, num_classes=n_classes)
    x_test = to_categorical(x_test, num_classes=n_classes)

    return TrainingDatasets(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )


def plot_history(history: History):
    # plot everything
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


class GestureModel:
    def __init__(self, load_model_path: str | None = None):
        self.model = create_model()
        if load_model_path:
            self.model.load_weights(load_model_path)

    def train(self, datasets: TrainingDatasets):
        # Training: Optimizer's Learning Rate
        learning_rate = 0.001

        # We use Adam to optimize a multi-class classification task
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Start the training

        # ...with an existing validation set:
        history = self.model.fit(
            datasets.x_train,
            datasets.x_test,
            validation_data=(datasets.y_train, datasets.y_test),
            epochs=100,
            batch_size=32,
        )

        return history

    def plot(self):
        plot_model(self.model, to_file="model.png", show_shapes=True)

    def save_weights(self, save_to: str):
        self.model.save_weights(save_to)

    def categorize(self, frame: SkeletalHGRFrame):

        predicted_prob = self.model.predict(frame.to_frame())
        predicted_class = predicted_prob.argmax(axis=-1).tolist()

        return predicted_class

    def evaluate_accuracy(self, datasets: TrainingDatasets):
        _, train_accuracy = self.model.evaluate(datasets.x_train, datasets.y_train)
        _, test_accuracy = self.model.evaluate(datasets.x_test, datasets.y_test)

        return (train_accuracy, test_accuracy)
