# digits

`digits` is a hand-gesture recognition toolkit built on top of the [Leap Motion]
hand-tracking controller.

It is based on https://github.com/guillaumephd/deep_learning_hand_gesture_recognition, which is in turn based on this paper:

G. Devineau, F. Moutarde, W. Xi and J. Yang, "Deep Learning for Hand Gesture Recognition on Skeletal Data," 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), Xi'an, China, 2018, pp. 106-113, doi: 10.1109/FG.2018.00025

It has three parts that work together:

1. A **C++ binary** (`digits`) and supporting shared libraries that connect to the
   Leap Motion service and stream hand-tracking frames, either as full JSON
   documents or as a flat 66-float "model" feature vector per hand.
2. A **C shared library** (`libdigitspy`) that implements an interface to Leap from Python
3. A **Python package** (`digipy`) that consumes those frames using a reusable numpy buffer to reduce allocations.

The recognition model is a multi-channel 1D CNN (see `digipy/training.py`) that
classifies a sliding window of 50 frames × 66 features into one of a small set of
gestures (see `digipy/gestures.py`).

## End-to-end workflow

1. **Build the C++ side:**
    ```sh
    cd src && ./build.sh && cd ..
    ```
2. **Capture data** (full JSON) and **label** it:
    ```sh
    ./src/build/digits --format full > digipy/capture.json
    cd digipy
    uv run digipy label-hand capture.json -o ../training/data
    ```
3. **Train** the model:
    ```sh
    uv run digipy train -t ../training/data -o ./model.weights.h5
    ```
4. **Run live inference**:
    ```sh
    uv run digipy inference -m ./model.weights.h5
    ```

## Prerequisites For Running

- A running **Leap Motion service** (the Leap Motion daemon/software). The C++
  binary and the Python live-inference path both talk to it over the Leap SDK.
- **Python ≥ 3.10** and [`uv`](https://docs.astral.sh/uv/) for the Python
  package (the project uses the `uv_build` backend).
- A built copy of `digits` and `libdigitspy` (see [Building](./docs/building.md))

## C++ binary (`digits`)

### What it does

`src/main.cpp` builds the `digits` executable. It registers a `Leap::Listener`
(`LibdigitsListener`) with a `Leap::Controller` and prints frames to stdout as
they arrive, throttled to a configurable interval. Two output formats are
supported via `--format`:

- `full` — one JSON object per frame describing every hand, finger, and bone
  (the format consumed by `digipy`'s capture/label/animate tools). See
  `logFrame()` in `src/lib/libdigits/logging.cpp`.
- `model` — one comma-separated line of **66 floats per hand**, the flat feature
  vector the gesture CNN expects (knuckle + tip positions for each of the five
  fingers, plus palm and wrist). See `modelInputs()` /
  `logModelFrame()` in the same file.

Flags (parsed in `src/lib/libdigits/cli.cpp`):

| Flag         | Default  | Meaning                                         |
| ------------ | -------- | ----------------------------------------------- |
| `--interval` | `50`     | Minimum milliseconds between emitted frames.    |
| `--format`   | `full`   | `full` (JSON) or `model` (CSV feature vectors). |
| `--output`   | `stdout` | The path to output samples to, or stdout.       |

### Running

```sh
./src/build/digits                      # full JSON, 50ms interval
./src/build/digits --format model       # 66-float CSV per hand
./src/build/digits --format full --interval 33
```

The process stays attached to the Leap service until you press **Enter**. Use the `--output`
flag to to capture a `capture.json` (JSONL) for later labeling or
animation, e.g.:

```sh
./src/build/digits --format full --output ./capture.json
```

### The `digipy` CLI

Find more information about digipy in [the docs](./docs/digipy.md).

The package provides a `digipy` console script (`digipy = "digipy:main"`):

```sh
uv run digipy <command> [options]
```

Subcommands:

- **`train`** — Train the gesture CNN on labeled samples under a data directory
  (default `../training/data`) and write weights to `model.weights.h5`.

    ```sh
    uv run digipy train -t ../training/data -o ./model.weights.h5
    ```

- **`inference`** — Load trained weights and run live gesture recognition. This
  calls `libdigits.start_listener()` and repeatedly feeds 50-frame windows
  through the model, printing the detected gesture and confidence for each
  window. Defaults: `--model ./model.weights.h5`, `--binary ../src/build/digits`,
  `--interval 50`, `--duration 50`.

    ```sh
    # run from digipy/ so the libdigitspy .so path resolves
    uv run digipy inference -m ./model.weights.h5
    ```

- **`animate-hand`** — Render a rotating 3D animation of a `capture.json`
  (JSONL) capture to an `.mp4`/`.gif`.

    ```sh
    uv run digipy animate-hand capture.json -o hand_animation.mp4 --fps 30
    ```

- **`label-hand`** — Open an interactive 3D viewer to scrub through a capture and
  export labeled training samples (`answer.txt` + `sample.json`) under an output
  directory (default `../training/data`).

    ```sh
    uv run digipy label-hand capture.json -o ../training/data
    ```

## Training data layout

Each labeled sample lives in its own subdirectory under `training/data/`
(e.g. `training/data/finger-gun/<sample>/`). A sample directory contains:

- `answer.txt` — the gesture label (a gesture name or integer, see
  `digipy/gestures.py`), and
- one or more `*.json` files, each a JSONL capture of exactly 50 frames.

`training/split.sh` chunks a raw JSONL capture into fixed-length `N.json` files
for use as samples:

```sh
./training/split.sh training/raw/capture.jsonl 50
```
