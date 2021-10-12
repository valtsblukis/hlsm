import torch

def unravel_spatial_arg(arg_flat, w, l, h):
    assert not isinstance(arg_flat, torch.Tensor) or arg_flat.dtype == torch.int64, "The following computations require int64 datatype to not break."
    x = arg_flat // (l * h)
    y = (arg_flat - x * l * h) //  h
    z = arg_flat - x * l * h - y * h
    return x, y, z


def ravel_spatial_arg(x, y, z, w, l, h):
    flat_coord = x * l * h + y * h + z
    return flat_coord


def spatial_argmax_3d(data):
    """
    Given a tensor of shape BxCxWxLxH where W,L,H are spatial dimensions,
    return a tensor of shape BxCx3 of 3D coordinates corresponding to the spatial argmax
    for each batch element, for each channel.
    """
    assert len(data.shape) == 3, "spatial_argmax_3d expected data of shape B x C x W x L x H"
    bs, c, w, l, h = data.shape
    data_flat = data.view([bs, c, -1])
    amax_flat = data_flat.argmax(2)
    x, y, z = unravel_spatial_arg(amax_flat, w, l, h)
    coords_3d = torch.stack([x, y, z], dim=2)
    return coords_3d