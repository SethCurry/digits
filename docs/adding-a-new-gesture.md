# Adding a New Gesture

This guide walks a new user through the whole process of teaching `digits` a
new hand gesture: registering the gesture, capturing and labeling training
data for it, retraining the recognition model, and testing it live with the
Leap Motion controller.

By the end you will have:

- a new entry in `digipy/src/digipy/gestures.py`,
- a directory of labeled training samples under `training/data/`,
- a freshly trained `model.weights.h5`, and
- a live recognizer that prints your gesture (and a confidence) as you hold
  it in front of the Leap.

> [!IMPORTANT]
> Every model must also be trained on a **`none` gesture** in addition to
> your real gestures. `none` is the "nothing is happening" class — see
> [Train the 'none' gesture](#train-the-none-gesture) below. Without it, the
> model has no idea what "not a gesture" looks like and will happily classify
> a relaxed or random hand as one of your gestures.

> [!NOTE]
> **A note on paths.** In this repository the C++ sources live in the
> `digits/` directory at the repo root. Some older docs (and a couple of
> defaults inside `digipy`) still refer to that directory as `src/`. This
> guide uses the paths as they exist on disk; wherever another doc disagrees,
> trust the commands here.

## Prerequisites

- A machine with a **Leap Motion controller** and the Leap Motion service
  (daemon) running. Every capture and the live test talk to this service.
- A C++17 compiler, CMake ≥ 3.24, and libxml2 — see the
  [building guide](./building.md).
- **Python ≥ 3.10** and [`uv`](https://docs.astral.sh/uv/).
- Roughly 30–60 minutes, most of it spent recording training data.

## On Window Sizing

You need to make sure that you use the same window size (how many frames you feed to the network at once)
through this entire process, and that you use the same interval between frames as well.

By default, digits uses a 50Hz clock rate and uses a window of 50 frames.
I.e. it records a frame every 20ms, and the LLM sees the past second
of frames at once.

You're free to change those, just make sure that you record your training at the same frame rate
you do inference at, and that your training samples are split into files according to the
window size.  A window size of 50 means 50 lines per file, 20 would be 20 lines, etc.

## How the pieces fit together

You don't need to know the internals to follow this guide, but three facts
explain most of the rules below:

1. **The model classifies fixed windows of frames.** One training sample is
   exactly **50 frames × 66 features** per frame (knuckle and tip positions
   for each finger, plus palm and wrist). At the default capture interval of
   50 ms, one window is **2.5 seconds** of hand motion.
2. **The set of gestures is an enum.** `digipy/src/digipy/gestures.py`
   defines the classes the model can output. The model's output-layer size is
   derived automatically from `len(Gesture)` (see the `N_GESTURES` constant in
   `digipy/src/digipy/training.py`), so adding a gesture is just an enum
   change — but it still means retraining from scratch, since the output
   layer changes size and old weight files no longer match it.
3. **Training data is a directory tree.** `digipy train` loads *every*
   directory under `training/data/`, so all gestures — old and new — are
   trained together in one run.

## Step 1 — Build everything

Build the C++ binary and shared libraries (from the repo root):

```sh
cd digits
./build.sh
```

This produces:

- `digits/build/digits` — the capture binary,
- `digits/build/lib/digitspy/liblibdigitspy.so` — the library used for live
  inference.

Then set up the Python package:

```sh
cd ../digipy
uv sync
```

Quick smoke test (Leap service must be running) — start the binary, hold a
hand over the controller, and confirm you see `Connected` on stderr and JSON
lines on stdout, then press **Enter** to quit:

```sh
cd ../        # repo root
./digits/build/digits --format full
```

## Step 2 — Add the gesture to `gestures.py`

Open `digipy/src/digipy/gestures.py` and add an enum member with the **next
free integer value**. For example, to add a thumbs-up gesture:

```python
class Gesture(Enum):
    NONE = 0
    FINGER_GUN = 1
    PEACE_SIGN = 2
    THUMBS_UP = 3   # <- new
```

The name you choose matters later: when labeling you'll type either the
snake_case enum name (`thumbs_up` — capitalization doesn't matter, the
underscores do) or the integer (`3`); both are accepted everywhere labels
are used.

The model's class count is derived from `len(Gesture)` automatically (see the
`N_GESTURES` constant in `digipy/src/digipy/training.py`, used as the default
for `create_model()` and `load_sample_data()`), so you do **not** need to
touch `training.py` — the output layer grows with the enum on its own.
Previously saved weights will no longer match the resized output layer, so
retraining from scratch is expected and automatic.

> Do **not** add a new entry for `none` — `Gesture.NONE` (value `0`) already
> exists. What `none` needs from you is *training data*, not an enum entry.

## Step 3 — Create training samples

### What a sample must look like

`digipy train` expects this layout under `training/data/`:

```
training/data/
  <any-directory-name>/      # e.g. thumbs-up — the name is arbitrary
    answer.txt               # the label: "thumbs_up" or "3"
    1.json                   # JSONL capture, exactly 50 frames
    2.json
    ...
```

Rules enforced by the loader (`load_data_dir` in `training.py`):

- `answer.txt` contains one label — a gesture name or integer (see Step 2).
- Each `*.json` file must contain **exactly 50 frames**, each with **exactly
  one hand** (frames with no hand are never logged by the capture binary;
  frames with two hands corrupt the count). Windows that don't match are
  **silently skipped** — if your sample counts come up short, this is why.
- Every 4th window (in load order) is held out for validation.

Aim for **at least ~20 windows per gesture** — the gestures shipped with the
repo have 20–38 each. More important than raw count is **variety**: record
the gesture at different positions and orientations in the Leap's field of
view.

### Capture

From the repo root, with the Leap service running, record a capture of the
new gesture:

```sh
./digits/build/digits --format full --output digipy/capture.json
```

(Or redirect stdout: `./digits/build/digits --format full > digipy/capture.json`.)

- Wait for `Connected` on stderr, hold your hand over the controller, and
  perform/hold the gesture. Keep **exactly one hand** in view.
- Leave the `--interval` flag at its default (50 ms) — that's what the
  training windows and the live inference path assume.
- Press **Enter** to stop recording.

Hold the pose long enough to get several 50-frame windows (each 2.5 s of
footage is one window). Repeat the capture a few times with the hand in
different positions. You can eyeball a capture before labeling it:

```sh
cd digipy
uv run digipy animate-hand capture.json -o ../hand_animation.mp4
```

You need one capture for **each** class, including a `none` capture — see
the next section.

### Label the capture

Pick one of the two workflows below.

**Option A — interactive (`label-hand`).** Scrub through the capture and
export windows one at a time:

```sh
cd digipy
uv run digipy label-hand capture.json -o ../training/data
```

- Type the label in the **label** box (`thumbs_up`) — the gesture must
  already exist in `gestures.py`, or saving fails with `unknown gesture`.
- **Set "save count" to `50`** — the box defaults to `20`, and non-50-frame
  windows are silently dropped by training.
- Scrub to the start of a clean hold, then click **Save next N frames** (or
  press `s`). Each save writes a new sample directory (auto-named
  `thumbs_up_<timestamp>`, or use the sample-name box).

**Option B — bulk split (`split.sh`).** If you recorded long, steady holds,
split the JSONL into 50-frame windows and write the label once:

```sh
cd ../        # repo root
mkdir -p training/data/thumbs-up
SPLIT_OUT_DIR=training/data/thumbs-up ./training/split.sh digipy/capture.json 50
echo "thumbs_up" > training/data/thumbs-up/answer.txt
```

Splitting is blind to content: the first and last windows will include the
moments you were getting into and out of the pose. Delete those files (e.g.
`rm training/data/thumbs-up/1.json`) so mislabeled windows don't pollute
the dataset.

### Train the 'none' gesture

The model is a softmax classifier: for **every** 2.5-second window it must
output *some* class, whether or not a gesture is being performed. `none`
(`Gesture.NONE = 0`) is that catch-all class, and it is already in the enum —
so "training the none gesture" means **recording windows where a hand is
visible but no known gesture is being performed**:

- a relaxed, open hand at rest,
- slow, random movement around the field of view,
- the transitions *into and out of* your other gestures,
- anything else that is *not* one of the gestures you're training.

Apply the same capture + labeling steps as above, with the label `none` (or
`0`). Give it at least as many windows as your real gestures, and favor
variety — the shipped `training/data/none-megalong/` directory is an example
of a `none` dataset. If you see the recognizer firing your new gesture on a
hand that's just sitting there, the fix is almost always *more and more
varied `none` samples*.

### Verify the dataset

Before training, check that every class is present and the windows are the
right size:

```sh
cd ../        # repo root
for d in training/data/*/; do
  echo "$d -> label '$(cat "$d/answer.txt")', $(ls "$d"/*.json | wc -l) windows"
done
```

Every directory's label should parse, and every gesture in the enum —
including `none` — should be represented. To spot-check window sizes:

```sh
wc -l training/data/thumbs-up/*.json | head
```

(Each file should be 50 lines.)

## Step 4 — Re-run training

From `digipy/`:

```sh
cd digipy
uv run digipy train -t ../training/data -o ./model.weights.h5
```

This loads *all* gesture directories (old and new), holds out every 4th
window for validation, trains for 100 epochs, and then plots train/validation
accuracy and loss. **Close the plot windows when you're done looking at
them** — the weights are written to `model.weights.h5` only after the plots
are closed.

What to look for in the plots:

- validation accuracy climbing into the high 90s (a 4-class problem with
  clean data usually gets there),
- validation loss coming down and staying near the training loss. If
  validation accuracy is high but live recognition is poor, your samples
  lack variety — go back to Step 3 and record more, especially for `none`.

Because the output layer changed size in Step 2, this is a from-scratch
training run; old weight files are not compatible with the new class count.

## Step 5 — Test

Run live inference with the freshly trained weights (from `digipy/`, with
the Leap service running):

```sh
DIGIPY_LIBDIGITS=../digits/build/lib/digitspy/liblibdigitspy.so \
uv run digipy inference -m ./model.weights.h5
```

> The `DIGIPY_LIBDIGITS` override is needed because `libdigits.py` defaults
> to a `../src/build/...` library path that doesn't exist in this tree. If
> you'd rather not type it every time, export the variable or update
> `_DEFAULT_LIB_PATH` in `digipy/src/digipy/libdigits.py`.

The listener buffers the most recent 50 frames; the first prediction appears
after the buffer fills (~2.5 s), and after that a prediction is printed for
each window as fast as the model can run, e.g.:

```
gesture detected  gesture=THUMBS_UP  confidence=0.99
```

Press **Ctrl+C** to quit. Things to try:

1. Hold your new gesture steady — expect its name with high confidence
   (~2.5 s after you settle into the pose).
2. Hold the pre-existing gestures (`finger-gun`, `peace-sign`) — they should
   still be recognized.
3. Relax your hand and wave it around — expect `NONE`.
4. Keep exactly one hand in view: frames with zero or two hands are not
   buffered, so the recognizer keeps repeating the last full window's
   prediction while your hand is absent or doubled.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `unknown gesture: ...` when labeling | The label isn't in `gestures.py` yet (add it first), or it isn't the enum name / integer — try `thumbs_up`, not `thumbs up` (underscores and spelling matter; capitalization doesn't). |
| Training crashes with an index/shape error | A label's integer value is outside the range of the enum, or you're loading old weights into a model whose output layer changed size when you added a gesture. Retrain from scratch after any enum change. |
| Fewer samples trained than you saved | Windows that aren't exactly 50 frames, or contain a two-hand frame, are silently skipped. Check with `wc -l` and re-record with one hand in view. |
| `label-hand` saved 20-frame samples | Change the **save count** box to `50` (its default is 20). |
| New gesture fires on relaxed/random hands | Not enough (or not enough varied) `none` data — record more `none` windows and retrain. |
| One class never predicted | Too few or too similar samples for that class; add variety (position, orientation, speed). |
| Inference can't load `liblibdigitspy.so` | Run from `digipy/` and set `DIGIPY_LIBDIGITS` as in Step 5 (or fix the default path in `libdigits.py`). |
| No `Connected` message / empty capture | The Leap Motion service isn't running or the device isn't streaming. |

## Checklist

- [ ] Build everything: `cd digits && ./build.sh`, then `cd ../digipy && uv sync`
- [ ] Add the gesture to `gestures.py` with the next free integer value (the model class count updates automatically from `len(Gesture)`)
- [ ] Capture and label ≥ ~20 windows for the new gesture (50 frames each, one hand in view)
- [ ] Capture and label a **`none`** gesture too — plus every other gesture in the enum
- [ ] Re-run training: `uv run digipy train -t ../training/data -o ./model.weights.h5`
- [ ] Test live: `uv run digipy inference -m ./model.weights.h5`
