"""Gesture labels used by the hand-tracking gesture model.

The set of gestures is no longer hard-coded here: it is defined by the
``gestures`` mapping of the configuration file (see :class:`digipy.cli.Config`,
loaded by ``load_config``).  Each key of that mapping is a gesture name, and
the position of a name in the config (in declaration order) is its class
index in the model.  This module only provides helpers that validate and
normalize gesture labels against such a configured list.
"""

from __future__ import annotations


def parse_gesture(label: str, gestures: list[str]) -> str:
    """Parse a gesture label against the configured *gestures* list.

    *gestures* is the list of gesture names from the configuration, in
    class-index order.  Accepts an integer index into that list (e.g.
    ``"1"``) or a gesture name (case-insensitive, e.g. ``"finger_gun"``),
    returning the canonical gesture name.  Raises ``ValueError`` for
    unknown values.
    """
    label = label.strip()

    # Integer strings are class indices into the configured gesture list
    # (the format used by answer.txt files of existing training samples).
    try:
        index = int(label)
    except ValueError:
        pass
    else:
        if 0 <= index < len(gestures):
            return gestures[index]
        raise ValueError(f"unknown gesture: {label!r}")

    for gesture in gestures:
        if gesture.lower() == label.lower():
            return gesture

    raise ValueError(f"unknown gesture: {label!r}")
