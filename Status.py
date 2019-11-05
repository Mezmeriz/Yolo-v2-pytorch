from enum import Enum


class TestStatus(Enum):
    UNTESTED = 0
    NOT_USED = 1
    FEATURE_INTERIOR = 2
    FEATURE_END = 3


class DebugStatus(Enum):
    NONE = 0
    START = 1
    CLOSE = 2
    NEAR = 3
