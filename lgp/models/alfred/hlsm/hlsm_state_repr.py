from typing import Iterable, Dict, List, Union, Tuple

import numpy as np
import torch
import math
from transforms3d import euler, affines
from lgp.abcd.repr.state_repr import StateRepr

from lgp.ops.misc import padded_roll_2d

from lgp.env.alfred.segmentation_definitions import intid_tensor_to_rgb, object_string_to_intid
import lgp.env.alfred.segmentation_definitions as segdef
from lgp.models.alfred.voxel_grid import VoxelGrid
from lgp.env.alfred.alfred_observation import AlfredObservation

from lgp.flags import TALL_GRID

# TODO: Compute these from voxelgrid origin and size
if TALL_GRID:
    GNDLVL = 0
    OBSLVL = 1
    CEILLVL = 8
else:
    GNDLVL = 2
    OBSLVL = 4  # (was 3)
    CEILLVL = 7


class AlfredSpatialStateRepr(StateRepr):
    """
    Represents world with:
        - self.data: BxCxHxWxL voxel grid
        - self.obs_mask: Bx1xHxWxL observability grid
        - self.inventory_vector: BxC agent inventory
    """
    def __init__(self,
                 data: VoxelGrid,
                 obs_mask: VoxelGrid,
                 vector: torch.tensor,
                 observation: Union[AlfredObservation, None]):
        super().__init__()
        self.data : VoxelGrid = data
        self.obs_mask : VoxelGrid = obs_mask
        self.inventory_vector : torch.tensor = vector
        self.observation : Union[AlfredObservation, None] = observation
        self.annotations = {}

    def to(self, device):
        data = self.data.to(device)
        obs_mask = self.obs_mask.to(device)
        inventory_vector = self.inventory_vector.to(device)
        observation = self.observation.to(device) if self.observation is not None else None
        if device == "cpu":
            inventory_vector = inventory_vector.float()
        return AlfredSpatialStateRepr(data, obs_mask, inventory_vector, observation)

    def __getitem__(self, item: int) -> "AlfredSpatialStateRepr":
        return AlfredSpatialStateRepr(self.data[item],
                                      self.obs_mask[item],
                                      self.inventory_vector[item],
                                      self.observation[item] if self.observation is not None else None)

    @classmethod
    def get_num_tensor_channels(cls):
        return cls.get_num_data_channels() + cls.get_num_vec_channels() + 1

    @classmethod
    def get_num_data_channels(cls):
        return segdef.get_num_objects()

    @classmethod
    def get_num_vec_channels(cls):
        return segdef.get_num_objects()

    def as_tensor(self):
        w = self.data.data.shape[2]
        l = self.data.data.shape[3]
        h = self.data.data.shape[4]
        # Spread out the vector across the spatial dimensions and concatenate to the spatial channels
        # Don't do the division - over a large voxel grid that might result in tiny tiny numbers
        expanded_vector = self.inventory_vector[:, :, None, None, None].repeat((1, 1, w, l, h))# / (w * l * h)
        concat = torch.cat([expanded_vector, self.data.data, self.obs_mask.data], dim=1)
        return concat

    @classmethod
    def from_tensor(cls, tensor):
        c_spatial = cls.get_num_data_channels()
        c_vec = cls.get_num_vec_channels()
        expanded_vector = tensor[:, :c_vec]
        vector = expanded_vector.mean(dim=2).mean(dim=2).mean(dim=2) # Average across spatial dimensions to obtain agent vector
        data = tensor[:, c_vec:-1, :, :, :]
        obs_mask = tensor[:, -1:, :, :, :]
        assert data.shape[1] == c_spatial, f"After slicing away agent vector, got wrong number of data channels: {data.shape[1]}, expected: {c_spatial}"
        return cls(data, obs_mask, vector, None)

    @classmethod
    def from_tensor_logits(cls, tensor):
        c_spatial = cls.get_num_data_channels()
        c_vec = cls.get_num_vec_channels()
        expanded_vector = tensor[:, :c_vec]
        vector = torch.sigmoid(expanded_vector.sum(dim=2).sum(dim=2)) # Average across spatial dimensions to obtain agent vector
        data = torch.sigmoid(tensor[:, c_vec:-1, :, :, :])
        obs_mask = torch.sigmoid(tensor[:, -1:, :, :, :])
        assert data.shape[1] == c_spatial, f"After slicing away agent vector, got wrong number of data channels: {data.shape[1]}, expected: {c_spatial}"
        return cls(data, obs_mask, vector, None)

    def get_obstacle_map_2d(self):
        # Consider "Wall" class to be obstacle even at ground level.
        occupancy_map_2d = self.data.occupancy[:, :, :, :, OBSLVL:].max(dim=4).values
        wall_id = object_string_to_intid("Wall")
        wall_map_2d = self.data.data[:, wall_id:wall_id+1, :, :, GNDLVL:CEILLVL].max(dim=4).values
        door_id = object_string_to_intid("Door")
        door_map_2d = self.data.data[:, door_id:door_id+1, :, :, GNDLVL:CEILLVL].max(dim=4).values
        window_id = object_string_to_intid("Window")
        window_map_2d = self.data.data[:, window_id:window_id+1, :, :, GNDLVL:CEILLVL].max(dim=4).values
        #floor_id = object_string_to_intid("Floor")
        #floor_map_2d = self.data.data[:, floor_id:floor_id+1, :, :, GNDLVL:OBSLVL].max(dim=4).values
        obstacle_map_2d = (occupancy_map_2d + wall_map_2d + door_map_2d + window_map_2d).clamp(0, 1)# - floor_map_2d).clamp(0, 1)
        #obstacle_map_2d = occupancy_map_2d
        return obstacle_map_2d

    @classmethod
    def get_2d_feature_dim(cls):
        return 7

    def center_2d_map_around_agent(self, grid, inverse=False):
        # Note: this function does NOT consider agent's rotation
        x, y, z = self.get_pos_xyz_vx()
        b, c, h, w = grid.shape
        cy = h // 2
        cx = w // 2
        sy = int(cy - y)
        sx = int(cx - x)
        # Shift back from centered around agent to global coodrinates
        if inverse:
            sy, sx = -sy, -sx
        grid = padded_roll_2d(grid, sy, sx)
        return grid

    def get_nav_features_2d(self, center_around_agent=False):
        obstacle_map = self.get_obstacle_map_2d()
        observability_map = self.get_observability_map_2d()

        pickable_ids = segdef.get_pickable_ids()
        recep_ids = segdef.get_receptacle_ids()
        toggle_ids = segdef.get_togglable_ids()
        openable_ids = segdef.get_openable_ids()
        ground_ids = segdef.get_ground_ids()

        pickable_map_2d = self.data.data[:, pickable_ids].max(dim=4).values.sum(dim=1, keepdim=True)
        recep_map_2d = self.data.data[:, recep_ids].max(dim=4).values.sum(dim=1, keepdim=True)
        toggle_map_2d = self.data.data[:, toggle_ids].max(dim=4).values.sum(dim=1, keepdim=True)
        open_map_2d = self.data.data[:, openable_ids].max(dim=4).values.sum(dim=1, keepdim=True)
        ground_map_2d = self.data.data[:, ground_ids].max(dim=4).values.sum(dim=1, keepdim=True)

        features_2d = torch.cat([observability_map, obstacle_map, pickable_map_2d, recep_map_2d, toggle_map_2d, open_map_2d, ground_map_2d], dim=1)

        if center_around_agent:
            features_2d = self.center_2d_map_around_agent(features_2d)

        features_2d = features_2d.float()
        return features_2d

    def viz_nav_features_2d(self, nav_features_2d):
        device = nav_features_2d.device
        colors = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]], device=device)
        nav_features_2d_viz = (nav_features_2d[:, :, None, :, :] * colors[None, :, :, None, None]).sum(dim=1) / 3
        return nav_features_2d_viz

    def get_nav_features_2d_viz(self, center_around_agent=False):
        features_2d = self.get_nav_features_2d(center_around_agent)
        return self.viz_nav_features_2d(features_2d)

    def get_observability_map_2d(self, floor_level=False, floor_only=False):
        if floor_level:
            observability_map_2d = self.obs_mask.data[:, :, :, :, GNDLVL:OBSLVL].max(dim=4).values
        elif floor_only:
            observability_map_2d = self.data.data[:, segdef.get_ground_ids(), :, :, GNDLVL:OBSLVL].max(dim=1, keepdim=True).values.max(dim=4).values
        else:
            observability_map_2d = self.obs_mask.data[:, :, :, :, OBSLVL:].max(dim=4).values
        return observability_map_2d

    def get_rpy(self) -> Tuple[float, float, float]:
        assert self.observation.pose.shape[0] == 1, "Only batch size 1 state reprs can return rpy"
        pose4f = self.observation.pose[0].detach().cpu().numpy()
        T, R, Z, S = affines.decompose44(pose4f)
        roll, pitch, yaw = euler.mat2euler(R)
        # TODO: Figure out why this seems to be needed to make the skill work out correctly:
        yaw = yaw + math.pi / 2
        return roll, pitch, yaw

    def get_camera_pitch_deg(self):
        return self.observation.cam_horizon_deg[0]

    def get_camera_pitch_rad(self):
        return math.radians(self.get_camera_pitch_deg())

    def get_camera_pitch_yaw(self):
        yaw = self.get_rpy()[2]
        pitch = self.get_camera_pitch_rad()
        return pitch, yaw

    def get_pos_xyz_m(self) -> Tuple[float, float, float]:
        assert self.observation.pose.shape[0] == 1, "Only batch size 1 state reprs can return a position"
        #pose4f = self.observation.pose[0].detach().cpu().numpy()
        #T, R, Z, S = affines.decompose44(pose4f)
        #x, y, z = T[0], T[1], T[2]

        p = self.observation.pose.detach().cpu()
        x = p[:, 0, 3].item()
        y = p[:, 1, 3].item()
        z = p[:, 2, 3].item()

        # TODO: Figure out why this seems to be needed to make the skill work out correctly:
        #y = -y
        x = -x
        return x, y, z

    def get_pos_m(self):
        # Grab correct entries from the 4x4 pose matrix
        x = self.observation.pose[:, 0, 3]
        y = self.observation.pose[:, 1, 3]
        z = self.observation.pose[:, 2, 3]

        # TODO: Figure out:
        y = -y
        vec = torch.stack([x,y,z], dim=1)
        return vec

    def get_agent_pos_m(self):
        agent_pos = self.observation.get_agent_pos()
        return agent_pos

    def get_pos_xyz_vx(self) -> Tuple[float, float, float]:
        agent_pos_m = self.get_agent_pos_m().detach().cpu()
        x_m = agent_pos_m[0].item()
        y_m = agent_pos_m[1].item()
        z_m = agent_pos_m[2].item()
        #x_m, y_m, z_m = self.get_pos_xyz_m()
        x_vx = (x_m - self.data.origin[0, 0]) / self.data.voxel_size
        y_vx = (y_m - self.data.origin[0, 1]) / self.data.voxel_size
        z_vx = (z_m - self.data.origin[0, 2]) / self.data.voxel_size
        return x_vx, y_vx, z_vx

    def get_origin_xyz_vx(self) -> Tuple[float, float, float]:
        x_vx = int((0 - self.data.origin[0, 0]) / self.data.voxel_size)
        y_vx = int((0 - self.data.origin[0, 1]) / self.data.voxel_size)
        z_vx = int((0 - self.data.origin[0, 2]) / self.data.voxel_size)
        return x_vx, y_vx, z_vx

    def get_agent_coord(self) -> torch.tensor:
        # TODO: Implement if needed
        raise NotImplementedError()

    @classmethod
    def new(cls, b, c, h, w, c2, device="cpu") -> "AlfredSpatialStateRepr":
        data = torch.zeros((b, c, h, w), device=device)
        obs_mask = torch.zeros((b, 1, h, w), device=device)
        vector = torch.zeros((b, c2), device=device)
        return cls(data, obs_mask, vector, None)

    @classmethod
    def collate(cls, states: Iterable["AlfredSpatialStateRepr"]) -> "AlfredSpatialStateRepr":
        """
        Creates a single Action that represents a batch of actions
        """
        datas = VoxelGrid.collate([s.data for s in states])
        obs_masks = VoxelGrid.collate([s.obs_mask for s in states])
        vectors = torch.cat([s.inventory_vector for s in states], dim=0)
        observation = (AlfredObservation.collate([s.observation for s in states])
                       if next(iter(states)).observation is not None else None)
        return cls(datas, obs_masks, vectors, observation)

    def view_voxel_map(self):
        import lgp.utils.render3d as r3d
        rgb_voxelgrid = self.make_rgb_voxelgrid(False)
        r3d.view_voxel_grid(rgb_voxelgrid)

    def make_rgb_voxelgrid(self, observability=False):
        if observability:
            rgb_data = 1 - self.obs_mask.data.repeat((1, 3, 1, 1, 1))
            occupancy = self.obs_mask.occupancy
        else:
            # Compute color at each voxel by a weighted average of colors of objects at that voxel
            if self.data.data.shape[1] > 3:
                rgb_data = intid_tensor_to_rgb(self.data.data)
            # Render colors as-is in the voxel grid
            else:
                rgb_data = self.data.data
            occupancy = self.data.occupancy

        # Wrap again in a VoxelGrid
        rgb_voxelgrid = VoxelGrid(rgb_data, occupancy, self.data.voxel_size, self.data.origin)
        return rgb_voxelgrid

    def represent_as_image(self, topdown2d=True, inventory=True) -> torch.tensor:
        return self.represent_as_image_with_inventory(topdown2d=topdown2d, inventory=inventory)

    def represent_as_image_with_inventory(self,
                                          animate=False,
                                          observability=False,
                                          topdown2d=False,
                                          inventory=True) -> List[np.ndarray]:
        if topdown2d:
            occupancy = self.data.occupancy.float()
            heights = torch.arange(0, occupancy.shape[4], device=occupancy.device)[None, None, None, None, :]
            grab_idx = (occupancy * heights).max(4, keepdim=True).values.long()
            grab_idx = grab_idx.repeat((1, self.data.data.shape[1], 1, 1, 1))
            grab_2d_data = self.data.data.float().gather(dim=4, index=grab_idx)

            #proj2d_data = self.data.data.max(4).values
            rgb_data = intid_tensor_to_rgb(grab_2d_data.float())
            image = rgb_data[:, :, :, :, 0]

            if inventory:
                vec_rgb = intid_tensor_to_rgb(self.inventory_vector.float())
                image[:, :, image.shape[2]-1, image.shape[3]-1] = vec_rgb
            return image
        else:
            import lgp.utils.render3d as r3d
            rgb_voxelgrid = self.make_rgb_voxelgrid(observability)
            image_or_images = r3d.render_voxel_grid(rgb_voxelgrid, animate=animate)

            if animate:
                return [image[np.newaxis, :, :, :] for image in image_or_images]
            else:
                return image_or_images[np.newaxis, :, :, :]
