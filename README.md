# digits

`digits` is a hand-gesture recognition toolkit built on top of the [Leap Motion]
hand-tracking controller. It has two parts that work together:

1. A **C++ binary** (`digits`) and supporting shared libraries that connect to the
   Leap Motion service and stream hand-tracking frames, either as full JSON
   documents or as a flat 66-float "model" feature vector per hand.
2. A **Python package** (`digipy`) that consumes those frames — directly through
   a `ctypes` binding to the C++ library for live inference, or from captured
   JSONL files for training, labeling, and animation.

The recognition model is a multi-channel 1D CNN (see `digipy/training.py`) that
classifies a sliding window of 50 frames × 66 features into one of a small set of
gestures (see `digipy/gestures.py`).

```
digits/
├── src/                 # C++ binary + libraries (Leap Motion SDK)
│   ├── main.cpp         # `digits` executable
│   ├── build.sh         # convenience build script (cmake)
│   ├── include/         # json.hpp used by the logging code
│   └── lib/
│       ├── leap/        # vendored Leap Motion SDK (headers + .so)
│       ├── libdigits/   # core listener / logging / CLI parsing
│       └── libdigitspy/ # C ABI binding for Python (ctypes)
├── digipy/              # Python package (uv project)
│   ├── pyproject.toml
│   └── src/digipy/
│       ├── libdigits.py     # ctypes wrapper around libdigitspy
│       ├── cli.py           # `digipy` CLI entry point
│       ├── training.py      # CNN model definition + training loop
│       ├── gestures.py      # Gesture enum
│       ├── capture.py       # JSONL frame parsing
│       ├── label_hand.py    # interactive 3D viewer / sample exporter
│       └── animate_hand.py  # 3D animation renderer
└── training/            # labeled training data
    ├── data/
    └── split.sh         # helper to chunk raw JSONL into samples
```

## Prerequisites

- A running **Leap Motion service** (the Leap Motion daemon/software). The C++
  binary and the Python live-inference path both talk to it over the Leap SDK.
- A C++17 compiler and **CMake** (≥ 3.24) for building the binary.
- **Python ≥ 3.10** and [`uv`](https://docs.astral.sh/uv/) for the Python
  package (the project uses the `uv_build` backend).

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

| Flag          | Default | Meaning                                                            |
|---------------|---------|--------------------------------------------------------------------|
| `--interval`  | `50`    | Minimum milliseconds between emitted frames.                       |
| `--format`    | `full`  | `full` (JSON) or `model` (CSV feature vectors).                    |

### Building

The included `src/build.sh` wipes and rebuilds `src/build/`:

```sh
cd src
./build.sh
```

This runs `cmake -B ./build && cmake --build ./build`. The resulting binary is
`src/build/digits`, and the shared libraries are
`src/build/lib/libdigits/liblibdigits.so` and
`src/build/lib/libdigitspy/liblibdigitspy.so`.

### Running

```sh
./src/build/digits                      # full JSON, 50ms interval
./src/build/digits --format model       # 66-float CSV per hand
./src/build/digits --format full --interval 33
```

The process stays attached to the Leap service until you press **Enter**. Pipe
stdout to a file to capture a `capture.json` (JSONL) for later labeling or
animation, e.g.:

```sh
./src/build/digits --format full > capture.json
```

## Python bindings (`digipy`)

### How the binding works

The C++ library `libdigitspy` (in `src/lib/libdigitspy/`) exposes a small C ABI
intended for FFI:

```c
void start_listener(int interval, int duration);   // interval in ms
void get_frames(float **dest);                      // fills a 50×66 buffer
```

Its listener buffers the most recent 50 model-format frames in a ring and signals
when the buffer is full (`LibdigitspyListener::full`). `get_frames()` blocks
until the buffer is full, then copies all 50 rows (66 floats each) into the
caller-supplied array of row pointers.

`digipy/src/digipy/libdigits.py` loads `liblibdigitspy.so` via `ctypes` and
exposes two functions:

```python
import digipy.libdigits as libdigits

libdigits.start_listener(interval_ms=20, duration=50)  # returns immediately
frames = libdigits.get_frames()                         # blocks until 50 frames buffered
# frames.shape == (50, 66), dtype float32
```

It allocates a single contiguous `(50 * 66)` ctypes float buffer and builds the
array-of-row-pointers the C side expects, so `get_frames()` returns a NumPy
`(50, 66)` view with no extra copying. **Note:** `libdigits.py` currently hard
-codes the shared-object path to
`"../src/build/lib/libdigitspy/liblibdigitspy.so"` (relative to the current
working directory), so it must be run from the `digipy/` directory after the C++
build, or that path must be edited.

### Installing `digipy`

```sh
cd digipy
uv sync          # create venv and install dependencies (tensorflow, numpy, ...)
```

### The `digipy` CLI

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

[Leap Motion]: https://developer.leapmotion.com/