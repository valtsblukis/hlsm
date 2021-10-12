# TODO: This entire file is almost a duplicate of env.alfred.alfred_action.py. Represent the common stuff accordingly!
from typing import Union, List
from lgp.abcd.subgoal import Subgoal
from lgp.abcd.task import Task
from lgp.ops.spatial_ops import unravel_spatial_arg

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_observation import AlfredObservation

import lgp.env.alfred.segmentation_definitions as segdef
from lgp.env.alfred.segmentation_definitions import OBJECT_INT_TO_STR

from lgp.models.alfred.projection.voxel_mask_to_image_mask import VoxelMaskToImageMask
from lgp.models.alfred.projection.image_to_voxels import ImageToVoxels
from lgp.models.alfred.voxel_grid import VoxelGrid

from lgp.ops.misc import index_to_onehot

from lgp.utils.viz import show_image
from lgp.flags import GLOBAL_VIZ

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

MASK_THRESHOLD = 0.3


class AlfredSubgoal(Subgoal, Task):
    class MakeActionException(Exception):
        ...

    """
    This is similar to AlfredAction, but the argument_mask is an allocentric voxel grid
    """
    def __init__(self,
                 action_type: torch.tensor,
                 argument_vector : torch.tensor,
                 argument_mask : Union[torch.tensor, None] = None):
        super().__init__()
        """
        action_type: B-length vector of integers indicating action type
        argument_vector: BxN matrix of one-hot argument vectors
        argument_mask: BxWxLxH tensor of masks indicating interaction positions
        """
        self.action_type = action_type
        # TODO: This includes an extra channel that represents "no argument"
        # That needs to be accounted for when converting to argument types!!
        self.argument_vector = argument_vector
        self.argument_mask = argument_mask

    def to(self, device):
        return AlfredSubgoal(self.action_type.to(device),
                             self.argument_vector.to(device),
                             self.argument_mask.to(device) if self.argument_mask is not None else None)

    def type_id(self):
        assert len(self.action_type) == 1, "Only single actions can have a single ID"
        return self.action_type[0].item()

    @classmethod
    def collate(cls, lst : List["AlfredSubgoal"]) -> "AlfredSubgoal":
        act_types = [l.action_type for l in lst]
        act_types = torch.cat(act_types, dim=0)
        arg_vectors = [l.argument_vector for l in lst]
        arg_vectors = torch.cat(arg_vectors, dim=0)
        arg_masks = [l.argument_mask for l in lst]

        for i, arg_mask in enumerate(arg_masks):
            if arg_mask is None:
                arg_masks[i] = VoxelGrid.create_empty(device=arg_masks[0].data.device)
        arg_masks = VoxelGrid.collate(arg_masks)
        return AlfredSubgoal(act_types, arg_vectors, arg_masks)

    def disperse(self) -> List["AlfredSubgoal"]:
        out = []
        for i in range(len(self.action_type)):
            out.append(AlfredSubgoal(
                self.action_type[i:i+1],
                self.argument_vector[i:i+1],
                self.argument_mask[i:i+1] if self.argument_mask[i] is not None else None))
        return out

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

    @classmethod
    def from_type_str_and_arg_vector(cls, type_str, arg_vec):
        type_id = cls.action_type_str_to_intid(type_str)
        type_vec = torch.tensor([type_id], device=arg_vec.device)
        return AlfredSubgoal(type_vec, arg_vec)

    @classmethod
    def from_type_str_arg_id_with_mask(cls, type_str, arg_id, mask):
        a = AlfredSubgoal.from_type_str_and_arg_id(type_str, arg_id)
        a.argument_mask = mask
        return a

    @classmethod
    def from_type_and_arg_id(cls, type_id, arg_id):
        type_vec = torch.tensor([type_id])
        argvec = torch.zeros([cls.get_action_arg_space_dim()])
        argvec[arg_id + 1] = 1 # TODO: Careful here. arg_id is obj_id
        argvec = argvec[None, :]
        return cls(type_vec, argvec)

    @classmethod
    def from_type_str_and_arg_id(cls, type_str, arg_id):
        type_id = cls.action_type_str_to_intid(type_str)
        return cls.from_type_and_arg_id(type_id, arg_id)

    @classmethod
    def extract_touch_argument(cls, action, observation):
        # B x C x W x L x H
        # state_repr = batch["states"][t]
        if action.argument_mask is None:
            return -1  # TODO: Check what to do here actually.
        semantic_image = observation.semantic_image[0]
        masked_semantics = action.argument_mask[None, :, :].to(semantic_image.device) * semantic_image
        semantic_vector = masked_semantics.sum(1).sum(1)
        argclass = semantic_vector.argmax().item()
        return argclass

    @classmethod
    def from_action_and_observation(cls, action: AlfredAction, observation: AlfredObservation):
        # Action type
        type_str = action.type_str()
        # Argument class
        arg_id = cls.extract_touch_argument(action, observation)
        # Spatial argument mask
        argument_mask_2d = action.argument_mask
        if argument_mask_2d is not None:
            depth_image = observation.depth_image
            hfov_deg = observation.hfov_deg
            extrinsics4f = observation.pose
            argument_mask_3d: VoxelGrid = ImageToVoxels()(
                action.argument_mask[None, None, :, :].float().to(depth_image.device), depth_image, extrinsics4f,
                hfov_deg)
        else:
            # TODO: Also allow creating "Stop" subgoals maybe?
            raise ValueError("Subgoals can only be created from interaction actions")
            argument_mask_3d = VoxelGrid.create_empty(1, 1, data_dtype=torch.float)

        return cls.from_type_str_arg_id_with_mask(type_str, arg_id, argument_mask_3d)

    @classmethod
    def get_action_type_space_dim(cls) -> int:
        return len(ACTION_TYPE_TO_IDX)

    @classmethod
    def get_action_arg_space_dim(cls) -> int:
        return segdef.get_num_objects() + 1

    @classmethod
    def action_type_str_to_intid(cls, action_type_str : str) -> int:
        return ACTION_TYPE_TO_IDX[action_type_str]

    @classmethod
    def action_type_intid_to_str(cls, action_type_intid : int) -> str:
        return IDX_TO_ACTION_TYPE[action_type_intid]

    def to_tensor(self, device="cpu", dtype=torch.int64):
        "Returns a Bx2 tensor where [0][0] is type id from 0 to 13, and [0][1] is arg_intid from -1 to 123"
        if len(self.action_type) == 1:
            type_id = self.type_intid()
            arg_id = self.arg_intid()
            return torch.tensor([[type_id, arg_id]], device=device, dtype=dtype)
        else:
            type_id = self.action_type[:, None]
            arg_id = self.argument_vector.argmax(dim=1, keepdim=True) - 1
            return torch.cat([type_id, arg_id], dim=1)

    def object_vector(self):
        return self.argument_vector[:, 1:]

    def type_intid(self):
        return int(self.action_type.item())

    def arg_intid(self):
        """
        Returns integer between -1 and 123, where -1 means no argument
        """
        # TODO: Batch support
        return int(self.argument_vector.argmax(dim=1)) - 1

    def type_str(self):
        if len(self.action_type) == 1:
            return self.action_type_intid_to_str(self.action_type[0].item())
        else:
            return [self.action_type_intid_to_str(i.item()) for i in self.action_type]

    def arg_str(self):
        intid = self.arg_intid()
        if intid == -1:
            return "NIL"
        elif intid >= len(OBJECT_INT_TO_STR):
            return "OutOfBounds"
        else:
            objid = intid
            return OBJECT_INT_TO_STR[objid]

    def type_oh(self):
        types = self.action_type
        oh = index_to_onehot(types, self.get_action_type_space_dim())
        return oh

    def arg_oh(self):
        return self.argument_vector

    def has_spatial_arg(self):
        return self.type_str() in INTERACT_ACTION_TYPES

    def get_argument_mask(self) -> torch.tensor:
        return self.argument_mask.data.data

    def build_spatial_arg_proposal(self, state_repr: "AlfredSpatialStateRepr"):
        #raise Exception("Should this actually be called?")
        #assert self.argument_mask is None, "Why build arg proposal if mask is already there?"
        # Locate the desired object in the voxel grid and produce a mask
        state_grid = state_repr.data.data
        # The 0th channel in argument_vector corresponds to the "no argument"
        spatial_argument = torch.einsum("bc,bcwlh->bwlh", self.argument_vector[:, 1:], state_grid.type(self.argument_vector.dtype))
        spatial_argument = spatial_argument[:, None, :, :, :]
        return spatial_argument

    def to_action(self, state_repr, observation: AlfredObservation, return_intermediates=False) -> AlfredAction:
        assert self.action_type.shape[0] == 1, "Can't convert a batch to action"

        if self.argument_mask is None:
            self.argument_mask = self.build_spatial_arg_proposal(state_repr)

        voxels_to_image2 = VoxelMaskToImageMask()
        action_type_str = self.type_str()
        argument_voxelgrid = self.argument_mask
        #argument_voxelgrid = VoxelGrid(
        #    self.argument_mask, self.argument_mask, state_repr.data.voxel_size, state_repr.data.origin)

        # Which pixels land within the selected voxel:
        fpv_voxel_argument_mask_f = voxels_to_image2(
            voxel_grid = argument_voxelgrid,
            extrinsics4f = observation.pose,
            depth_image = observation.depth_image,
            hfov_deg = observation.hfov_deg
        )
        fpv_voxel_argument_mask = fpv_voxel_argument_mask_f / (torch.max(fpv_voxel_argument_mask_f) + 1e-10)

        # Which pixels match the class of the object based on segmentation image:
        fpv_semantic_match_mask = torch.einsum("bc,bchw->bhw", self.object_vector(), observation.semantic_image.float())
        fpv_semantic_match_mask = fpv_semantic_match_mask / (torch.max(fpv_semantic_match_mask) + 1e-10)

        # Take pixels that agree with both class and voxel masks
        fpv_argument_mask = fpv_voxel_argument_mask * fpv_semantic_match_mask[:, None, :, :]
        fpv_argument_mask = (fpv_argument_mask > MASK_THRESHOLD).float()

        if GLOBAL_VIZ:
            show_image(fpv_argument_mask[0], "fpv_argument_mask", scale=1, waitkey=1)
            show_image(fpv_voxel_argument_mask[0], "fpv_VOXEL_argument_mask", scale=1, waitkey=1)
            show_image(fpv_semantic_match_mask[0], "fpv_SEMANTC_argument_mask", scale=1, waitkey=1)

        # Alfred actions use 2D numpy masks
        fpv_argument_mask = fpv_argument_mask[0, 0]

        intermediates = {
            "fpv_argument_mask": fpv_argument_mask[None, None, :, :],
            "fpv_voxel_argument_mask": fpv_voxel_argument_mask,
            "fpv_semantic_argument_mask": fpv_semantic_match_mask[:, None, :, :]
        }

        if return_intermediates:
            return AlfredAction(action_type_str, fpv_argument_mask), intermediates
        else:
            return AlfredAction(action_type_str, fpv_argument_mask)

    # TODO: Check usages of this
    def get_argmax_spatial_arg_pos_xyz_vx(self):#, state_repr : Union["AlfredSpatialStateRepr", None]=None):
        # Returns the 3D spatial position of the most likely point in the argument mask
        ab, ac = self.argument_vector.shape
        assert ab == 1, "Only batch size of 1 supported so far"
        #state_grid = state_repr.data.data
        #sb, sc, w, l, h = state_grid.shape
        #assert sc == ac - 1, "Semantic vector needs to match voxel grid number of channels"
        #assert ab == sb, "Semantic vector needs the same batch size as state voxel grid"

        #spatial_argument = self.build_spatial_arg_proposal(state_repr)
        b, _, w, l, h = self.argument_mask.data.shape

        # Find the coordinates of the most matching object in the environment
        argmask_0 = self.argument_mask.data[0].view([1, -1])
        pos = argmask_0.argmax(dim=1)
        x, y, z = unravel_spatial_arg(pos, w, l, h)
        return x.item(), y.item(), z.item()

    def is_stop(self):
        return self.type_str() == "Stop"

    def __eq__(self, other: "AlfredSubgoal"):
        # TODO: Consider if we need to add mask here as well
        if not isinstance(other, AlfredSubgoal):
            return False
        # Explore and Stop actions don't depend on the action argument
        return self.action_type == other.action_type and (
            (not self.has_spatial_arg())
                        or
            (self.argument_vector == other.argument_vector).all())

    def __str__(self):
        return f"HLA: {self.type_str()} : {self.arg_str()}"
