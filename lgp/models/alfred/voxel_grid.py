from typing import Iterable, List
import torch
import numpy

from lgp.models.alfred.projection.constants import ROUNDING_OFFSET
from lgp.flags import TALL_GRID

if TALL_GRID:
    class DefaultGridParameters:
        GRID_SIZE_X = 15.25
        GRID_SIZE_Y = 15.25
        GRID_SIZE_Z = 2.5
        GRID_RES = 0.25
        GRID_ORIGIN = [-7.625, -7.625, -0.125]
        # The start is off by 0.125 so that agent navigates along gridcell centers
else:
    class DefaultGridParameters:
        GRID_SIZE_X = 15.25
        GRID_SIZE_Y = 15.25
        GRID_SIZE_Z = 2.0
        GRID_RES = 0.25
        GRID_ORIGIN = [-7.625, -7.625, -0.5]
        # The start is off by 0.125 so that agent navigates along gridcell centers


class VoxelGrid:
    def __init__(self, data, occupancy, voxel_size, origin):
        self.data = data
        self.occupancy = occupancy
        self.voxel_size = voxel_size
        self.origin = origin

    def to(self, device):
        data = self.data.to(device)
        occupancy = self.occupancy.to(device)
        origin = self.origin.to(device)

        # Keep data in float format in CPU and half format on GPU to save on GPU memory
        if device == "cpu":
            data = data.float()
            occupancy = occupancy.float()
            origin = origin.float()
        if device == "gpu":
            data = data.half()
            occupancy = occupancy.half()

        return VoxelGrid(data, occupancy, self.voxel_size, origin)

    def __getitem__(self, i):
        return VoxelGrid(self.data[i], self.occupancy[i], self.voxel_size, self.origin[i])

    def get_integer_bounds(self):
        d = self.data.device
        b, c, w, l, h = self.data.shape
        min_bounds = torch.tensor([0, 0, 0], device=d)
        max_bounds = torch.tensor([w, l, h], device=d)
        return min_bounds, max_bounds

    def get_centroid_coord_grid(self):
        b, c, w, l, h = self.data.shape
        device = self.data.device

        # TODO: Assume all origins in the batch are the same
        # Voxel coordinates are taken as the center coordinates of each voxel
        xrng = torch.arange(0, w, device=device).float() * self.voxel_size + self.origin[0, 0] + self.voxel_size * (0.5 - ROUNDING_OFFSET)
        yrng = torch.arange(0, l, device=device).float() * self.voxel_size + self.origin[0, 1] + self.voxel_size * (0.5 - ROUNDING_OFFSET)
        zrng = torch.arange(0, h, device=device).float() * self.voxel_size + self.origin[0, 2] + self.voxel_size * (0.5 - ROUNDING_OFFSET)

        xrng = xrng[:, None, None].repeat((1, l, h))
        yrng = yrng[None, :, None].repeat((w, 1, h))
        zrng = zrng[None, None, :].repeat((w, l, 1))

        grid = torch.stack([xrng, yrng, zrng]).unsqueeze(0)
        return grid

    @classmethod
    def default_parameters(cls):
        return DefaultGridParameters()

    @classmethod
    def create_from_mask(cls, mask, params=DefaultGridParameters()):
        voxel_size = params.GRID_RES
        origin = torch.tensor([params.GRID_ORIGIN], device=mask.device)
        return cls(mask, mask, voxel_size, origin)

    @classmethod
    def create_empty(cls, batch_size=1, channels=1, params=DefaultGridParameters(), device="cpu", data_dtype=torch.float32):
        w = int(params.GRID_SIZE_X / params.GRID_RES)
        l = int(params.GRID_SIZE_Y / params.GRID_RES)
        h = int(params.GRID_SIZE_Z / params.GRID_RES)
        data = torch.zeros([batch_size, channels, w, l, h], device=device, dtype=data_dtype)
        occupancy = torch.zeros([batch_size, 1, w, l, h], device=device, dtype=data_dtype)
        voxel_size = params.GRID_RES
        origin = torch.tensor([params.GRID_ORIGIN], device=device)
        return cls(data, occupancy, voxel_size, origin)

    @classmethod
    def collate(cls, voxel_grids : List["VoxelGrid"]):
        device = voxel_grids[0].data.device
        datas = torch.cat([v.data.to(device) for v in voxel_grids], dim=0)
        occupancies = torch.cat([v.occupancy.to(device) for v in voxel_grids], dim=0)
        voxel_size = next(iter(voxel_grids)).voxel_size
        origins = torch.cat([v.origin.to(device) for v in voxel_grids], dim=0)
        return VoxelGrid(datas, occupancies, voxel_size, origins)
