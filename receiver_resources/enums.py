from enum import Enum, auto


class RecurrenceType(Enum):
    ONCE_PER_MONTH = auto()
    BY_DAY_OF_MONTH = auto()
    BY_DAY_OF_WEEK = auto()
    RANDOM_INFREQUENT = auto()
    RANDOM = auto()


class ValueConsistency(Enum):
    EXTREMELY_CONSISTENT = auto()
    RELATIVELY_CONSISTENT = auto()
    SOME_VARIANCE = auto()
    VERY_VARIANT = auto()