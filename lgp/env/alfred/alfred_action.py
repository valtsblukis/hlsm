from typing import Union
from lgp.abcd.action import Action

import numpy as np
import torch

IDX_TO_ACTION_TYPE = {
    0: "RotateLeft",
    1: "RotateRight",
    2: "MoveAhead",
    3: "LookUp",
    4: "LookDown",
    5: "OpenObject",
    6: "CloseObject",
    7: "PickupObject",
    8: "PutObject",
    9: "ToggleObjectOn",
    10: "ToggleObjectOff",
    11: "SliceObject",
    12: "Stop"
}
# TODO: Reinstate Stop action as an action type

ACTION_TYPE_TO_IDX = {v:k for k,v in IDX_TO_ACTION_TYPE.items()}
ACTION_TYPES = [IDX_TO_ACTION_TYPE[i] for i in range(len(IDX_TO_ACTION_TYPE))]

NAV_ACTION_TYPES = [
    "RotateLeft",
    "RotateRight",
    "MoveAhead",
    "LookUp",
    "LookDown"
]

INTERACT_ACTION_TYPES = [
    "OpenObject",
    "CloseObject",
    "PickupObject",
    "PutObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject"
]

class AlfredAction(Action):
    def __init__(self,
                 action_type: str,
                 argument_mask : torch.tensor):
        super().__init__()
        self.action_type = action_type
        self.argument_mask = argument_mask

    def to(self, device):
        self.argument_mask = self.argument_mask.to(device) if self.argument_mask is not None else None
        return self

    @classmethod
    def stop_action(cls):
        return cls("Stop", cls.get_empty_argument_mask())

    @classmethod
    def get_empty_argument_mask(cls) -> torch.tensor:
        return torch.zeros((300, 300))

    @classmethod
    def get_action_type_space_dim(cls) -> int:
        return len(ACTION_TYPE_TO_IDX)

    @classmethod
    def action_type_str_to_intid(cls, action_type_str : str) -> int:
        return ACTION_TYPE_TO_IDX[action_type_str]

    @classmethod
    def action_type_intid_to_str(cls, action_type_intid : int) -> str:
        return IDX_TO_ACTION_TYPE[action_type_intid]

    @classmethod
    def get_interact_action_list(cls):
        return INTERACT_ACTION_TYPES

    @classmethod
    def get_nav_action_list(cls):
        return NAV_ACTION_TYPES

    def is_valid(self):
        if self.action_type in NAV_ACTION_TYPES:
            return True
        elif self.argument_mask is None:
            print("AlfredAction::is_valid -> missing argument mask")
            return False
        elif self.argument_mask.sum() < 1:
            print("AlfredAction::is_valid -> empty argument mask")
            return False
        return True

    def type_intid(self):
        return self.action_type_str_to_intid(self.action_type)

    def type_str(self):
        return self.action_type

    def to_alfred_api(self) -> (str, Union[None, np.ndarray]):
        if self.action_type in NAV_ACTION_TYPES:
            argmask_np = None
        else: # Interaction action needs a mask
            if self.argument_mask is not None:
                if isinstance(self.argument_mask, torch.Tensor):
                    argmask_np = self.argument_mask.detach().cpu().numpy()
                else:
                    argmask_np = self.argument_mask
            else:
                argmask_np = None
        return self.action_type, argmask_np

    def is_stop(self):
        return self.action_type == "Stop"

    def __eq__(self, other: "AlfredAction"):
        return self.action_type == other.action_type and self.argument_mask == other.argument_mask

    def __str__(self):
        return f"AA: {self.action_type}"

    def represent_as_image(self):
        if self.argument_mask is None:
            return torch.zeros((1, 300, 300))
        else:
            return self.argument_mask