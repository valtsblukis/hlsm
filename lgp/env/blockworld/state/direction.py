from enum import Enum


class Direction(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    UP = "UP"

    def __str__(self):
        return str(self.name)

    def to_vector(self):
        if self == Direction.LEFT:
            return (0, -1)
        if self == Direction.RIGHT:
            return (0, 1)
        if self == Direction.UP:
            return (-1, 0)
        if self == Direction.DOWN:
            return (1, 0)

    @classmethod
    def from_vector(cls, vector):
        if vector == (0, -1):
            return Direction.LEFT
        if vector == (0, 1):
            return Direction.RIGHT
        if vector == (-1, 0):
            return Direction.UP
        if vector == (1, 0):
            return Direction.DOWN
        raise ValueError(f"Can't create blockworld direction for vector: {vector}")