import itertools
import copy
import math
import random
from typing import List

import torch

from lgp.env.blockworld import config as config
from lgp.env.blockworld.bwobservation import BwObservation
from lgp.env.privileged_info import PrivilegedInfo
from lgp.env.blockworld.state.item import Item
from lgp.env.blockworld.state.room import Room
from lgp.env.blockworld.state.visuals import one_hot_to_image


class World:

    def __init__(self, rooms: List[Room], agent_room_idx: int, inventory: List[Item]):
        self.rooms = rooms
        self.agent_room_idx = agent_room_idx
        self.inventory = inventory
        self.stopped = False

    @classmethod
    def make_random(cls):
        # Make a grid of random rooms
        rooms = [Room.make_random((i, j))
                 for (i, j) in itertools.product(range(config.NUM_ROOMS),
                                                 range(config.NUM_ROOMS))]

        # Iterate over pairs of neighboring rooms and connect them
        for room1, room2 in itertools.product(rooms, rooms):
            # Skip non-neighboring rooms
            if room1 is room2 or not room1.is_physically_adjacent(room2):
                continue
            # With some probability, connect the rooms
            if random.random() < config.CONNECT_P:
                room1.add_door(room2)

        # Randomly drop agent in one of the rooms
        agent_room_idx = random.randrange(len(rooms))

        # Initially start with an empty inventory
        inventory = []

        return cls(rooms, agent_room_idx, inventory)

    def __eq__(self, other: "World"):
        return (self.inventory == other.inventory) and (self.agent_room_idx == other.agent_room_idx) and (self.rooms == other.rooms) and (self.stopped == other.stopped)

    def _get_grid_size(self):
        return config.NUM_ROOMS * self._get_room_size()

    def _get_room_size(self):
        # Each room is a NxN grid, surrounded by an additional 2 exterior walls
        return config.ROOM_SIZE + 2

    def get_current_room(self) -> Room:
        return self.rooms[self.agent_room_idx]

    def place_in_inventory(self, item: Item):
        self.inventory.append(item)

    def move_agent_to_room(self, new_room_coord: (int, int)):
        matching_rooms = [i for i, r in enumerate(self.rooms) if r.coord == new_room_coord]
        assert len(matching_rooms) > 0, f"Attempted to move agent to a room {new_room_coord} which doesn't exist!"
        new_room_idx = matching_rooms[0]
        self.agent_room_idx = new_room_idx

    def get_one_hot_world_tensor(self) -> torch.tensor:
        """
        Creates a tensor CxNxN size tensor representation of the blockworld environment, where
         N: (config.ROOM_SIZE + 2) * config.NUM_ROOMS
         C: 2 * <number of colors> + 3
        At each spatial location, the cell is represented by a vector that is a concatenation of:
         - Agent vector: one-hot vector with 3 properties: walkable, agent present, observed
         - Item vector: one-hot vector identifying the color of the item. All zeros if no item is there.
         - Room vector: one-hot vector identifying the color of the room
        :return:
        """
        grid_size = self._get_grid_size()
        room_size = self._get_room_size()
        c_name_to_idx = config.get_spatial_state_name_to_idx()
        tensor_grid = torch.zeros((len(c_name_to_idx), grid_size, grid_size))

        # Draw all the rooms on the grid
        for i, room in enumerate(self.rooms):
            t = room.coord[0] * room_size
            l = room.coord[1] * room_size
            d = t + room_size
            r = l + room_size
            tensor_grid[:, t:d, l:r] = room.get_one_hot_tensor()

            # If agent is in this room, mark it in the center of the room
            if self.agent_room_idx == i:
                cy = t + math.floor(room_size / 2)
                cx = l + math.floor(room_size / 2)
                tensor_grid[c_name_to_idx["agent"], cy, cx] = 1.0
        return tensor_grid

    def get_one_hot_agent_tensor(self):
        c_name_to_idx = config.get_spatial_state_name_to_idx()

        agent_vector = torch.zeros((len(c_name_to_idx) + 1, ))
        for item in self.inventory:
            agent_vector[c_name_to_idx[f"item-{item.color}"]] += 1
        agent_vector[-1] = 1 if self.stopped else 0
        return agent_vector

    def get_observability_mask(self, full_observability : bool = False) -> torch.tensor:
        """
        :return: 1xDxD tensor indicating the room the agent is in with ones + additional 1 cell on each side of the room.
         The rest of the cells are filled with zeros.
        """
        grid_size = self._get_grid_size()
        room_size = self._get_room_size()

        if full_observability:
            mask = torch.ones((1, grid_size, grid_size))
        else:
            mask = torch.zeros((1, grid_size, grid_size))
            agent_room_coord = self.rooms[self.agent_room_idx].coord
            viz_t = agent_room_coord[0] * room_size - 1
            viz_b = viz_t + room_size + 2
            viz_l = agent_room_coord[1] * room_size - 1
            viz_r = viz_l + room_size + 2
            viz_l, viz_t = max(0, viz_l), max(0, viz_t)
            viz_r, viz_b = min(grid_size, viz_r), min(grid_size, viz_b)
            mask[:, viz_t:viz_b, viz_l:viz_r] = 1.0
        return mask

    def get_observation(self, full_observability : bool = False) -> torch.tensor:
        """
        :return: CxDxD tensor that is the state tensor, masked by the observability mask
        """
        state_tensor = self.get_one_hot_world_tensor().unsqueeze(0)
        observability_mask = self.get_observability_mask(full_observability).unsqueeze(0)
        observation_tensor = state_tensor * observability_mask
        agent_vector = self.get_one_hot_agent_tensor().unsqueeze(0)
        agent_coordinate = [self.rooms[self.agent_room_idx].get_center_coordinate()]
        privileged_info = PrivilegedInfo([copy.deepcopy(self)])

        observation = BwObservation(observation_tensor, observability_mask, agent_vector, agent_coordinate, privileged_info)
        return observation

    def represent_as_image(self) -> torch.tensor:
        tensor_repr = self.get_one_hot_world_tensor()
        return one_hot_to_image(tensor_repr)


# Quick main function to test state generation
from lgp.utils.utils import tensorshow
if __name__ == "__main__":
    # Test generating random world
    print("Creating a random world state!")
    world = World.make_random()
    print("Building image representation")
    img_repr = world.represent_as_image()
    print("Displaying")
    tensorshow("test_world", img_repr, scale=16, waitkey=0, normalize=False)
    print("Done!")