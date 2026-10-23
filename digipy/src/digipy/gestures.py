from enum import Enum


class Gesture(Enum):
    NONE = 1
    CONDUCTOR_UP = 2
    CONDUCTOR_DOWN = 3


def parse_gesture(name: str):
    if name == "none":
        return Gesture.NONE

    if name == "conductor_up":
        return Gesture.CONDUCTOR_UP
    if name == "conductor_down":
        return Gesture.CONDUCTOR_DOWN

    raise ValueError("unknown gesture: " + name)
