from typing import Tuple


class Item:
    def __init__(self, color: str, coord: Tuple[int, int]):
        self.color = color
        self.coord = tuple(coord)

    def __eq__(self, other):
        return (self.color == other.color) and (self.coord == other.coord)

    def __str__(self):
        return f"Item of color={self.color} at coord={self.coord}"