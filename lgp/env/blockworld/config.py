from typing import Tuple
import numpy as np
from colour import Color


FULL_OBSERVABILITY = False

DEFAULT_HORIZON = 20

NUM_ROOMS = 3
ROOM_SIZE = 5

CONNECT_P = 0.5
ITEM_COUNT_POISSON_LAMBDA = 1.0

COLORS = ["red", "green", "orange", "pink", "blue"]
AGENT_COLOR = "purple"
WALL_COLOR = "gray"
WALKABLE_COLOR = "white"


assert ROOM_SIZE % 2 == 1, "Only odd room sizes supported right now!"

class ChannelDefinition:
    def __init__(self, idx, name, color):
        self.idx = idx
        self.name = name
        self.color = color

def _build_state_vector_channel_definition():
    """
    :return: a List of ChannelDefinition objects for this environment
    """
    idx = 0
    channels = []
    channels.append(ChannelDefinition(idx, "agent", Color(AGENT_COLOR))); idx += 1
    channels.append(ChannelDefinition(idx, "wall", Color(WALL_COLOR))); idx += 1
    channels.append(ChannelDefinition(idx, "walkable", Color(WALKABLE_COLOR))); idx += 1
    for i in range(len(COLORS)):
        channels.append(ChannelDefinition(idx, f"item-{COLORS[i]}", Color(COLORS[i]))); idx += 1
    for i in range(len(COLORS)):
        # Make rooms a lighter shade by blending with white
        room_color_rgb = 0.2 * np.asarray(Color(COLORS[i]).get_rgb()) + 0.8 * np.asarray(Color("white").get_rgb())
        room_color = Color(rgb=room_color_rgb)
        channels.append(ChannelDefinition(idx, f"room-{COLORS[i]}", room_color)); idx += 1
    return channels


def get_state_vector_channel_definitions():
    return _build_state_vector_channel_definition()


def get_spatial_state_name_to_idx():
    definitions = _build_state_vector_channel_definition()
    c_name_to_idx = {d.name: d.idx for d in definitions}
    return c_name_to_idx

def get_spatial_state_idx_to_name():
    definitions = _build_state_vector_channel_definition()
    c_idx_to_name = {d.idx: d.name for d in definitions}
    return c_idx_to_name

def get_spatial_state_idx_to_color():
    definitions = _build_state_vector_channel_definition()
    c_idx_to_color = {d.idx: d.color for d in definitions}
    return c_idx_to_color

def get_grid_size():
    return (ROOM_SIZE + 2) * NUM_ROOMS

# TODO: In general, the coordinate should probably be treated more consistently e.g. with coordinate frames
def agent_to_room_item_coord(item_coord: Tuple[int, int]) -> Tuple[int, int]:
    return (item_coord[0] + int(ROOM_SIZE / 2),
            item_coord[1] + int(ROOM_SIZE / 2))

def room_to_agent_item_coord(item_coord: Tuple[int, int]) -> Tuple[int, int]:
    return (item_coord[0] - int(ROOM_SIZE / 2),
            item_coord[1] - int(ROOM_SIZE / 2))