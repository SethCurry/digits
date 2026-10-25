"""Thin ctypes wrapper around the ``libdigitspy`` shared library.

The C library streams hand-tracking frames into a fixed-size sliding window
of ``N_FRAMES`` rows by ``N_INPUTS`` channels.  :func:`get_frames` returns
that window as a ``(N_FRAMES, N_INPUTS)`` :class:`numpy.ndarray` view into a
reusable contiguous buffer (no per-call allocation).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

# Number of frames in the sliding window and channels per frame.  These must
# match the model's input shape (see digipy.training.create_model).
N_FRAMES = 50
N_INPUTS = 66

# Path to the shared library.  Override with the DIGIPY_LIBDIGITS env var.
_DEFAULT_LIB_PATH = "../digits/build/lib/digitspy/liblibdigitspy.so"
_LIB_PATH = Path(os.environ.get("DIGIPY_LIBDIGITS", _DEFAULT_LIB_PATH))
libdigits = ctypes.CDLL(str(_LIB_PATH))

# A single contiguous buffer holding all frames back-to-back.  This lets us
# hand back a single numpy ndarray at the end.
RAW_FRAMES = (ctypes.c_float * (N_FRAMES * N_INPUTS))()


# The C side expects ``float **dest``: an array of pointers, one per row, each
# pointing at a buffer of N_INPUTS floats.  Build that array of row pointers
# into the contiguous buffer above.
def _build_row_ptrs(raw_frames: ctypes.Array, n_frames: int, n_inputs: int):
    ptrs = (ctypes.POINTER(ctypes.c_float) * n_frames)()
    row_stride = n_inputs * ctypes.sizeof(ctypes.c_float)
    base = ctypes.addressof(raw_frames)
    for i in range(n_frames):
        ptrs[i] = ctypes.cast(
            base + i * row_stride,
            ctypes.POINTER(ctypes.c_float),
        )
    return ptrs


ROW_PTRS = _build_row_ptrs(RAW_FRAMES, N_FRAMES, N_INPUTS)

_FLOAT_PTR_PTR = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

libdigits.get_frames.argtypes = [_FLOAT_PTR_PTR]
libdigits.start_listener.argtypes = [ctypes.c_int, ctypes.c_int]


def start_listener(interval_ms: int, duration: int) -> None:
    """Start streaming frames from the digits binary."""
    libdigits.start_listener(interval_ms, duration)


def get_frames() -> np.ndarray:
    """Return the current sliding window as a ``(N_FRAMES, N_INPUTS)`` array.

    The returned array is a view into the shared ``RAW_FRAMES`` buffer, so it
    is overwritten on the next call to :func:`get_frames`.  Copy it if you need
    to keep it.
    """
    libdigits.get_frames(ROW_PTRS)
    # NOTE: numpy.ctypeslib.as_array ignores the `shape` argument when handed a
    # flat ctypes array, returning a 1-D (N_FRAMES * N_INPUTS,) array.  Build
    # the (N_FRAMES, N_INPUTS) view explicitly from the raw buffer.
    return np.frombuffer(RAW_FRAMES, dtype=np.float32).reshape((N_FRAMES, N_INPUTS))
