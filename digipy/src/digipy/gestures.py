"""Gesture labels used by the hand-tracking gesture model."""

from enum import Enum


class Gesture(Enum):
    NONE = 0
    FINGER_GUN = 1
    PEACE_SIGN = 2


# Lowercase name -> Gesture, for parsing human-friendly labels.
_GESTURE_BY_NAME = {g.name.lower(): g for g in Gesture}


def parse_gesture(name: str) -> Gesture:
    """Parse a gesture label from either an integer string or a name.

    Accepts an integer (e.g. ``"1"``) or a lowercase gesture name
    (e.g. ``"finger_gun"``).  Raises ``ValueError`` for unknown values.
    """
    try:
        return Gesture(int(name))
    except ValueError:
        pass

    try:
        return _GESTURE_BY_NAME[name.lower()]
    except KeyError:
        raise ValueError(f"unknown gesture: {name!r}") from None
