import math
import random
from typing import Tuple, List, Union

import numpy as np
import torch

from lgp.env.blockworld import config as config
from lgp.env.blockworld.state.direction import Direction
from lgp.env.blockworld.state.item import Item


class Door:
    def __init__(self, source: "Room", target: "Room", direction: Direction):
        self.source = source
        self.target = target
        self.direction = direction

    def __eq__(self, other: Union["Door", Direction]):
        if isinstance(other, Direction):
            return self.direction == other
        else:
            # Can't check source and target rooms, because that would give stack overflow
            return (self.direction == other.direction) and (self.source.coord == other.source.coord) and (self.target.coord == other.target.coord)

    def __str__(self):
        print(f"Door with direction: {self.direction} from {self.source.coord} to {self.target.coord}")

    def __hash__(self):
        return hash(str(self))

class Room:
    def __init__(self, coord: Tuple[int, int], color: str, items: List[Item], doors: List[Door]):
        self.coord = tuple(coord)
        self.color = color
        self.items = items
        self.doors = doors

    @classmethod
    def make_random(cls, coord: Tuple[int, int]):
        size = config.ROOM_SIZE
        # Sample room color
        color = random.choice(config.COLORS)

        # Sample number of items
        num_items = min(np.random.poisson(lam=config.ITEM_COUNT_POISSON_LAMBDA), (size - 1) ** 2)

        # Sample item colors: object of the same color as room is twice as likely
        item_color_pdf = np.asarray([3 if c == color else 1 for c in config.COLORS])
        item_color_probs = item_color_pdf / (item_color_pdf.sum() + 1e-10)
        item_colors = np.random.choice(config.COLORS, size=num_items, p=item_color_probs)

        # Sample item coordinates (the entire room size, leaving horizontal and vertical corridors in the center)
        cell_coords = cls.get_all_item_cell_coords()
        #dim_coords = np.hstack([np.arange(0, (size-1)/2), np.arange(math.ceil(size/2), size)]).astype(np.int32)
        #cell_coords = np.stack(np.meshgrid(dim_coords, dim_coords), axis=2).reshape((-1, 2))
        item_coords = cell_coords[np.random.choice(len(cell_coords), num_items, replace=False).tolist()].tolist()

        items = [Item(col, coord) for col, coord in zip(item_colors, item_coords)]

        # Initially, don't create any doors
        doors = []
        return cls(coord, color, items, doors)

    @classmethod
    def get_all_item_cell_coords(cls):
        size = config.ROOM_SIZE
        dim_coords = np.hstack([np.arange(0, (size-1)/2), np.arange(math.ceil(size/2), size)]).astype(np.int32)
        cell_coords = np.stack(np.meshgrid(dim_coords, dim_coords), axis=2).reshape((-1, 2))
        return cell_coords

    def __eq__(self, other: "Room"):
        return (self.coord == other.coord) and (self.color == other.color) and (self.items == other.items) and (self.doors == other.doors)

    def __hash__(self):
        return hash(str(self.coord) + str(self.color) + str(self.items) + str(self.doors))

    def push_item(self, item: Item):
        overlapping_items = [i for i, x in enumerate(self.items) if x.coord[0] == item.coord[0] and x.coord[1] == item.coord[1]]
        # If the item would overlap another item, ignore it
        if len(overlapping_items) > 0:
            return False
        self.items.append(item)
        return True

    def pop_item(self, coord: Tuple[int, int]) -> Union[Item, None]:
        matching_items = [i for i, x in enumerate(self.items) if x.coord[0] == coord[0] and x.coord[1] == coord[1]]
        if len(matching_items) == 0:
            return None
        item_idx = matching_items[0]
        # Retrieve the selected item
        item = self.items[item_idx]
        # Delete the selected item
        self.items = self.items[:item_idx] + self.items[item_idx+1:]
        return item

    def is_physically_adjacent(self, other: "Room"):
        l1dist = np.linalg.norm(np.asarray(self.coord) - np.asarray(other.coord), ord=1)
        return l1dist <= 1.0

    def add_door(self, other: "Room"):
        assert self.is_physically_adjacent(other), "Can't connect rooms that are not adjacent"
        if self.coord[1] < other.coord[1]:
            self.doors.append(Door(source=self, target=other, direction=Direction.RIGHT))
            other.doors.append(Door(source=other, target=self, direction=Direction.LEFT))
        elif self.coord[1] > other.coord[1]:
            self.doors.append(Door(source=self, target=other, direction=Direction.LEFT))
            other.doors.append(Door(source=other, target=self, direction=Direction.RIGHT))
        elif self.coord[0] < other.coord[0]:
            self.doors.append(Door(source=self, target=other, direction=Direction.DOWN))
            other.doors.append(Door(source=other, target=self, direction=Direction.UP))
        elif self.coord[0] > other.coord[0]:
            self.doors.append(Door(source=self, target=other, direction=Direction.UP))
            other.doors.append(Door(source=other, target=self, direction=Direction.DOWN))

    def get_center_coordinate(self):
        top_left_coord = tuple(map(lambda m: m * (config.ROOM_SIZE + 2), self.coord))
        center_coord = tuple(map(lambda m: m + int((config.ROOM_SIZE + 2) / 2), top_left_coord))
        return center_coord

    def get_one_hot_tensor(self):
        c_name_to_idx = config.get_spatial_state_name_to_idx()

        # Color the entire room
        size = config.ROOM_SIZE + 2
        tensor_repr = torch.zeros((len(c_name_to_idx), size, size))
        tensor_repr[c_name_to_idx[f"room-{self.color}"], :, :] = 1.0

        # Color each item
        for item in self.items:
            y = item.coord[0] + 1
            x = item.coord[1] + 1
            c = c_name_to_idx[f"item-{item.color}"]
            tensor_repr[c, y, x] = 1.0

        # Paint all four walls
        wall_idx = c_name_to_idx["wall"]
        gap_a = 2
        gap_b = size - 2
        tensor_repr[wall_idx][0, :] = 1.0
        tensor_repr[wall_idx][size - 1, :] = 1.0
        tensor_repr[wall_idx][:, 0] = 1.0
        tensor_repr[wall_idx][:, size - 1] = 1.0

        # Cut out doors and draw walking paths
        path_idx = c_name_to_idx["walkable"]
        center = math.floor((config.ROOM_SIZE + 2) / 2)
        if Direction.LEFT in self.doors:
            tensor_repr[wall_idx][gap_a:gap_b, 0] = 0.0
            tensor_repr[path_idx, center, :center+1] = 1.0
        if Direction.RIGHT in self.doors:
            tensor_repr[wall_idx][gap_a:gap_b, size-1] = 0.0
            tensor_repr[path_idx, center, center:] = 1.0
        if Direction.UP in self.doors:
            tensor_repr[wall_idx][0, gap_a:gap_b] = 0.0
            tensor_repr[path_idx, :center+1, center] = 1.0
        if Direction.DOWN in self.doors:
            tensor_repr[wall_idx][size-1, gap_a:gap_b] = 0.0
            tensor_repr[path_idx, center:, center] = 1.0

        # Make sure that we can't "see through walls" (e.g. seeing the room color under the walls)
        no_wall_mask = 1 - tensor_repr[wall_idx]
        room_channels = [idx for name, idx in c_name_to_idx.items() if name.startswith("room")]
        tensor_repr[room_channels] = tensor_repr[room_channels] * no_wall_mask

        return tensor_repr