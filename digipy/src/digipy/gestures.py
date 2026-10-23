from enum import Enum


class Gesture(Enum):
    NONE = 0
    FINGER_GUN = 1
    PEACE_SIGN = 2


def parse_gesture(name: str):
    try:
        name_int = int(name)
        return Gesture(name_int)
    except:
        if name == "none":
            return Gesture.NONE

        if name == "finger_gun":
            return Gesture.FINGER_GUN
        if name == "peace_sign":
            return Gesture.PEACE_SIGN

        raise ValueError("unknown gesture: " + name)
