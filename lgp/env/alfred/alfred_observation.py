from typing import Iterable, Union, List
import torch

from lgp.abcd.observation import Observation

from lgp.env.privileged_info import PrivilegedInfo

import lgp.env.alfred.segmentation_definitions as segdef


VISUALIZE_AUGMENTATIONS = False
from lgp.utils.viz import show_image


class AlfredObservation(Observation):
    def __init__(self,
                 rgb_image: torch.tensor,
                 depth_image: torch.tensor,
                 semantic_image: torch.tensor,
                 inventory_vector: torch.tensor,
                 pose: torch.tensor,
                 hfov_deg: float,
                 cam_horizon_deg: List[float],
                 privileged_info: Union[PrivilegedInfo, List[PrivilegedInfo]]):
        super().__init__()
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.semantic_image = semantic_image
        self.inventory_vector = inventory_vector
        self.pose = pose
        self.hfov_deg = hfov_deg
        self.cam_horizon_deg = cam_horizon_deg
        self.privileged_info = privileged_info
        self.error_causing_action = None
        self.last_action_error = False

        self.extra_rgb_frames = []

        # This is used only during test time. TODO: Make sure pose is correct and use it instead.
        self.agent_pos = None

    def __getitem__(self, item):
        rgb_image = self.rgb_image[item]
        depth_image = self.depth_image[item]
        semantic_image = self.semantic_image[item]
        inventory_vector = self.inventory_vector[item]
        pose = self.pose[item]
        hfov_deg = self.hfov_deg
        cam_horizon_deg = [self.cam_horizon_deg[item]]
        if isinstance(self.privileged_info, PrivilegedInfo):
            privileged_info = self.privileged_info
        else:
            privileged_info = self.privileged_info[item]
        assert self.error_causing_action is None, "Cannot treat observations with error_causing_action as batches"
        return AlfredObservation(
            rgb_image,
            depth_image,
            semantic_image,
            inventory_vector,
            pose,
            hfov_deg,
            cam_horizon_deg,
            privileged_info
        )

    def set_agent_pos(self, agent_pos):
        self.agent_pos = agent_pos

    def get_agent_pos(self, device="cpu"):
        # TODO: This is a temporary workaround. Figure it out and standardize it:
        if self.agent_pos is None:
            raise ValueError("Requesting agent_pos from observation, but set_agent_pos hasn't been called")
        return self.agent_pos.to(device)

    def get_depth_image(self):
        if isinstance(self.depth_image, torch.Tensor):
            return self.depth_image
        # depth image is a DepthEstimate
        else:
            return self.depth_image.get_trustworthy_depth()

    def get_objects_image(self):
        interactive_mask = self.semantic_image[:, segdef.TABLETOP_OBJECT_IDS, :, :].max(dim=1, keepdim=True).values
        return interactive_mask

    def is_compressed(self):
        return self.semantic_image.shape[1] == 1

    def compress(self):
        # If semantic image is in a one-hot representation, convert to a more space-saving integer representation
        if not self.is_compressed():
            # TODO: This doesn't support anything bigger than 128!!
            self.semantic_image = self.semantic_image.type(torch.uint8).argmax(dim=1, keepdim=True)

    def uncompress(self):
        # If semantic image is in an integer representation, convert it to a one-hot representation
        if self.is_compressed():
            n_c = segdef.get_num_objects()
            rng = torch.arange(0, n_c, 1, device=self.semantic_image.device, dtype=torch.uint8)
            onehotrepr = (rng[None, :, None, None] == self.semantic_image).type(torch.uint8)
            self.semantic_image = onehotrepr

    def data_augment(self):
        import lgp.env.alfred.alfred_observation_augmentation as aug
        was_compressed = self.is_compressed()
        self.uncompress()
        dtype = self.rgb_image.dtype
        self.rgb_image = self.rgb_image.float()
        if VISUALIZE_AUGMENTATIONS:
            show_image(self.rgb_image[0], "before_aug", waitkey=1, scale=1)
        # Perform augmentation on the RGB image
        # Apply random image color transformations per object instance class
        aug.seg_region_augmentation(self)
        # Randomly flip images horizontally
        aug.flip_augmentation(self)
        if VISUALIZE_AUGMENTATIONS:
            show_image(self.rgb_image[0], "after_aug", waitkey=True, scale=1)
        self.rgb_image = self.rgb_image.type(dtype)
        if was_compressed:
            self.compress()

    def set_error_causing_action(self, action):
        self.error_causing_action = action
        if action is not None:
            self.last_action_error = True

    def to(self, device) -> "AlfredObservation":
        obs_o = AlfredObservation(
            self.rgb_image.to(device),
            self.depth_image.to(device),
            self.semantic_image.to(device),
            self.inventory_vector.to(device),
            self.pose.to(device),
            self.hfov_deg,
            self.cam_horizon_deg,
            self.privileged_info
        )
        obs_o.error_causing_action = self.error_causing_action
        obs_o.last_action_error = self.last_action_error
        obs_o.agent_pos = self.agent_pos.to(device) if self.agent_pos is not None else None
        obs_o.extra_rgb_frames = [f.to(device) for f in self.extra_rgb_frames]
        return obs_o

    @classmethod
    def collate(cls, observations: Iterable["AlfredObservation"]) -> "AlfredObservation":
        rgb_images = torch.cat([o.rgb_image for o in observations], dim=0)
        depth_images = torch.cat([o.depth_image for o in observations], dim=0)
        semantic_images = torch.cat([o.semantic_image for o in observations], dim=0)
        inventory_vectors = torch.cat([o.inventory_vector for o in observations], dim=0)
        poses = torch.cat([o.pose for o in observations], dim=0)
        hfov_deg = next(iter(observations)).hfov_deg
        cam_horizon_deg = [o.cam_horizon_deg[0] for o in observations]
        privileged_infos = [o.privileged_info for o in observations]
        return AlfredObservation(
            rgb_image=rgb_images,
            depth_image=depth_images,
            semantic_image=semantic_images,
            inventory_vector=inventory_vectors,
            pose=poses,
            hfov_deg=hfov_deg,
            cam_horizon_deg=cam_horizon_deg,
            privileged_info=privileged_infos
        )

    def represent_as_image(self, semantic=True, rgb=True, depth=True, horizontal=False) -> torch.tensor:
        imglist = []
        if rgb:
            imglist.append(self.rgb_image)
        if semantic:
            was_compressed = False
            if self.is_compressed():
                was_compressed = True
                self.uncompress()
            imglist.append(segdef.intid_tensor_to_rgb(self.semantic_image))
            if was_compressed:
                self.compress()
        if depth:
            imglist.append(self.get_depth_image().repeat((1, 3, 1, 1)) / 5.0)
        return torch.cat(imglist, dim=3 if horizontal else 2)
