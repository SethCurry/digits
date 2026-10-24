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
