# TODO: This entire file is almost a duplicate of env.alfred.alfred_action.py. Represent the common stuff accordingly!
from typing import Union
from lgp.abcd.subgoal import Subgoal
from lgp.abcd.task import Task
from lgp.ops.spatial_ops import unravel_spatial_arg

import numpy as np
import torch


IDX_TO_ACTION_TYPE = {
    0: "OpenObject",
    1: "CloseObject",
    2: "PickupObject",
    3: "PutObject",
    4: "ToggleObjectOn",
    5: "ToggleObjectOff",
    6: "SliceObject",
    7: "Stop",
    8: "Explore"
}

ACTION_TYPE_TO_IDX = {v:k for k,v in IDX_TO_ACTION_TYPE.items()}
ACTION_TYPES = [IDX_TO_ACTION_TYPE[i] for i in range(len(IDX_TO_ACTION_TYPE))]

NAV_ACTION_TYPES = [
    "Explore"
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


class AlfredActionHL(Subgoal, Task):
    """
    This is similar to AlfredAction, but the argument_mask is an allocentric voxel grid
    """
    def __init__(self,
                 action_type: str,
                 argument_mask : torch.tensor,
                 argument_vector : Union[torch.tensor, None] = None):
        super().__init__()
        self.action_type = action_type
        self.argument_mask = argument_mask
        self.argument_vector = argument_vector

    def type_id(self):
        return self.action_type_str_to_intid(self.action_type)

    def to(self, device):
        self.argument_mask = self.argument_mask.to(device) if self.argument_mask is not None else None
        self.argument_vector = self.argument_vector.to(device) if self.argument_vector is not None else None
        return self

    @classmethod
    def get_action_type_space_dim(cls) -> int:
        return len(ACTION_TYPE_TO_IDX)

    @classmethod
    def action_type_str_to_intid(cls, action_type_str : str) -> int:
        return ACTION_TYPE_TO_IDX[action_type_str]

    @classmethod
    def action_type_intid_to_str(cls, action_type_intid : int) -> str:
        return IDX_TO_ACTION_TYPE[action_type_intid]

    def get_argmax_spatial_arg_pos_xyz_vx(self, state_repr=None):
        # Returns the 3D spatial position of the most likely point in the argument mask

        b, c, w, l, h = self.argument_mask.data.shape
        assert b == 1, "Only batch size of 1 supported so far"
        argmask_0 = self.argument_mask.data[0].view([c, -1])
        pos = argmask_0.argmax(dim=1)
        x, y, z = unravel_spatial_arg(pos, w, l, h)
        return x.item(), y.item(), z.item()

    @classmethod
    def get_2d_feature_dim(cls):
        return 2

    def get_spatial_arg_2d_features(self):
        CAMERA_HEIGHT = 1.576
        argmask_3d_vx = self.argument_mask
        vn = argmask_3d_vx.data.shape[4]
        vmin = argmask_3d_vx.origin[0, 2].item()
        vmax = vmin + vn * argmask_3d_vx.voxel_size
        vrange = torch.linspace(vmin, vmax, vn, device=self.argument_mask.data.device) - CAMERA_HEIGHT
        masked_vrange = argmask_3d_vx.data * vrange[None, None, None, None, :]
        argmask_2d_heights = masked_vrange.sum(dim=4) / (argmask_3d_vx.data.sum(dim=4) + 1e-3)
        argmask_2d_bool = argmask_3d_vx.data.max(dim=4).values
        argmask_2d = torch.cat([argmask_2d_bool, argmask_2d_heights], dim=1)
        return argmask_2d

    def type_intid(self):
        return self.action_type_str_to_intid(self.action_type)

    def type_str(self):
        return self.action_type

    def has_spatial_arg(self):
        return self.action_type in INTERACT_ACTION_TYPES

    def is_stop(self):
        return self.action_type == "Stop"

    def __eq__(self, other: "AlfredActionHL"):
        # Explore and Stop actions don't depend on the action argument
        return self.action_type == other.action_type and (
            (not self.has_spatial_arg())
                        or
            (self.argument_mask == other.argument_mask).all())

    def __str__(self):
        return f"AA: {self.action_type}"
