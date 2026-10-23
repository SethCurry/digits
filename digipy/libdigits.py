import ctypes

import numpy

libdigits = ctypes.CDLL("../src/build/libdigitspy.so")


FRAMES_POINTER = numpy.ctypeslib.ndpointer(dtype=numpy.float32, ndim=2, flags="C")

libdigits.get_frame.argtypes = [FRAMES_POINTER]
libdigits.start_listener.argtypes = [ctypes.c_int, ctypes.c_int]


def start_listener(interval_ms: int, duration: int):
    libdigits.start_listener(interval_ms, duration)
    return FRAMES_POINTER


def get_frames(frames: numpy.ndarray):
    libdigits.get_frames(frames)
