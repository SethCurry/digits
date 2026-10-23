import ctypes

import numpy

libdigits = ctypes.CDLL("../src/build/lib/libdigitspy/liblibdigitspy.so")

N_FRAMES = 50
N_INPUTS = 66

# A single contiguous buffer holding all frames back-to-back.
# This lets us hand back a single numpy ndarray at the end.
RAW_FRAMES = (ctypes.c_float * (N_FRAMES * N_INPUTS))()

# The C side expects `float **dest`: an array of pointers, one per row,
# each pointing at a buffer of N_INPUTS floats.  Build that array of row
# pointers into the contiguous buffer above.
ROW_PTRS = (ctypes.POINTER(ctypes.c_float) * N_FRAMES)()
_row_stride = N_INPUTS * ctypes.sizeof(ctypes.c_float)
for i in range(N_FRAMES):
    ROW_PTRS[i] = ctypes.cast(
        ctypes.addressof(RAW_FRAMES) + i * _row_stride,
        ctypes.POINTER(ctypes.c_float),
    )

FLOAT_PTR_PTR = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))

libdigits.get_frames.argtypes = [FLOAT_PTR_PTR]
libdigits.start_listener.argtypes = [ctypes.c_int, ctypes.c_int]


def start_listener(interval_ms: int, duration: int):
    libdigits.start_listener(interval_ms, duration)


def get_frames():
    libdigits.get_frames(ROW_PTRS)
    # NOTE: numpy.ctypeslib.as_array ignores the `shape` argument when handed a
    # flat ctypes array, so it would return a 1-D (N_FRAMES * N_INPUTS,) array.
    # Build the (N_FRAMES, N_INPUTS) view explicitly from the raw buffer.
    frames = numpy.frombuffer(RAW_FRAMES, dtype=numpy.float32).reshape(
        (N_FRAMES, N_INPUTS)
    )
    return frames