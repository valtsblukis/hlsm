from typing import Dict
import random
import itertools
import torch

from lgp.abcd.agent import Agent
from lgp.abcd.repr.state_repr import StateRepr

from lgp.env.alfred.alfred_observation import AlfredObservation
from lgp.env.alfred.tasks import AlfredTask
from lgp.env.alfred.alfred_action import AlfredAction, ACTION_TYPES, NAV_ACTION_TYPES, INTERACT_ACTION_TYPES

import lgp.env.alfred.segmentation_definitions as segdef


class RandomValidAgent(Agent):
    def __init__(self):
        super().__init__()
        ...

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def start_new_rollout(self, task: AlfredTask, state_repr: StateRepr = None):
        # Random agent doesn't care about the task
        ...

    def act(self, observation: AlfredObservation) -> AlfredAction:
        # First sample the type of action
        permissible_actions = ACTION_TYPES
        action_type = random.choice(permissible_actions)

        # Interaction actions need an object mask
        if action_type in INTERACT_ACTION_TYPES:
            objects_present = observation.semantic_image.max(2).values.max(2).values
            obj_ids = torch.arange(objects_present.shape[1])
            present_obj_ids = obj_ids[objects_present.bool()[0]]
            present_obj_ids = present_obj_ids.detach().cpu().numpy().tolist()
            present_interactive_obj_ids = [o for o in present_obj_ids if o in segdef.INTERACTIVE_OBJECT_IDS]
            if len(present_interactive_obj_ids) > 0:
                interact_obj_id = random.choice(present_interactive_obj_ids)
                interact_mask = observation.semantic_image[0, interact_obj_id, :, :]
            else:
                interact_mask = torch.zeros_like(observation.semantic_image[:, 0, :, :])
        else:
            interact_mask = None

        action = AlfredAction(action_type, interact_mask)
        return action