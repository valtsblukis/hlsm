from typing import Iterable
import torch

from lgp.abcd.observation import Observation

from lgp.env.privileged_info import PrivilegedInfo
from lgp.env.blockworld.state import visuals


class BwObservation(Observation):
    def __init__(self,
                 world_obs: torch.tensor,
                 obs_mask: torch.tensor,
                 agent_vector: torch.tensor,
                 agent_coordinate: torch.tensor,
                 privileged_info: PrivilegedInfo):
        """
        :param world_obs: BxCxDxD-dimensional tensor
        :param obs_mask: Bx1xDxD-dimensional tensor, indicating which spatial locations in world_obs are non-zero
        :param agent_vector: a BxC-dimensional vector of a one-hot encoding of the agent's inventory, and whether it stopped
        :param agent_coordinate: a Bx2D vector indicating the location of agent in the world_obs tensor
        """
        super().__init__()
        assert world_obs.shape[0] == obs_mask.shape[0], "Batch size of mask and observation must be the same."
        assert world_obs.shape[2] == obs_mask.shape[2], "Height of mask and observation must be the same."
        assert world_obs.shape[3] == obs_mask.shape[3], "Width of mask and observation must be the same."
        self.world_obs = world_obs
        self.obs_mask = obs_mask
        self.agent_vector = agent_vector
        self.agent_coordinate = agent_coordinate
        self.privileged_info = privileged_info

    def to(self, device) -> "BwObservation":
        self.world_obs = self.world_obs.to(device)
        self.obs_mask = self.obs_mask.to(device)
        self.agent_vector = self.agent_vector.to(device)
        return self

    def collate(cls, observations: Iterable["BwObservation"]) -> "BwObservation":
        raise NotImplementedError("Collating observations not yet implemented")

    def represent_as_image(self) -> torch.tensor:
        # Represent state as an RGB image
        state_image = visuals.one_hot_to_image(self.world_obs)
        batch_size = self.world_obs.shape[0]

        # Show inventory as an extra row on the bottom that accumulates items
        inventory_image = torch.zeros(batch_size, self.world_obs.shape[1], 5, self.world_obs.shape[3])
        for b in range(batch_size):
            for i, x in enumerate(self.agent_vector[b]):
                if x > 0:
                    inventory_image[b, i, 5-int(x):, i] = 1.0
        inventory_image = visuals.one_hot_to_image(inventory_image)

        # Concatenate state and inventory representations
        complete_image = torch.zeros((batch_size, state_image.shape[1], state_image.shape[2] + 1 + 5, state_image.shape[3]))
        complete_image[:, :, 0:state_image.shape[2], :] = state_image
        complete_image[:, :, state_image.shape[2]+1:, :] = inventory_image
        return complete_image
