import torch
import lgp.env.blockworld.config as config


def _draw_over(tensor_a, tensor_b, mask=None):
    """
    Draw the contents of tensor_b over the contents of tensor_a as if it were a canvas, replacing information
    that is already there at all pixels where tensor_b (or mask if provided) is non-zero.
    :return:
    """
    # Compute an alpha mask or tensor b
    if mask is None:
        mask = tensor_b.sum(1, keepdims=True) > 0
    mask = mask.float().clamp(0.0, 1.0)
    out = tensor_a * (1 - mask.type(torch.float32)) + tensor_b * mask
    return out

def one_hot_to_image(tensor_repr: torch.tensor) -> torch.tensor:
    """
    Maps a BxCxDxD-dimensional tensor with C semantic one-hot channels to a Bx3xDxD-dimensional RGB representation,
    using the colors defined for each channel in config.py
    :param tensor_repr: BxCxDxD-dimensional tensor representing the world
    :return: Bx3xDxD-dimensional tensor
    """
    with torch.no_grad():

        channel_defs = config.get_state_vector_channel_definitions()
        c_name_to_idx = config.get_spatial_state_name_to_idx()
        channel_color_tensor = torch.tensor([[c.color.rgb for c in channel_defs]], device=tensor_repr.device) # B x C x 3
        world_color_tensor = (channel_color_tensor[:, :, :, None, None] * tensor_repr[:, :, None, :, :]) # B x C x 3 x H x W

        room_channels = [d.idx for d in channel_defs if d.name.startswith("room")]
        item_channels = [d.idx for d in channel_defs if d.name.startswith("item")]

        # First paint the rooms
        image_repr = world_color_tensor[:, room_channels].sum(1) # Sum across semantic channels to obtain a color

        # Then paint the walls
        walls = world_color_tensor[:, c_name_to_idx["wall"]]
        walls_alpha = tensor_repr[:, c_name_to_idx["wall"]]
        image_repr = _draw_over(image_repr, walls, walls_alpha)
        # Paint the walkable paths
        paths = world_color_tensor[:, c_name_to_idx["walkable"]]
        paths_alpha = tensor_repr[:, c_name_to_idx["walkable"]]
        image_repr = _draw_over(image_repr, paths, paths_alpha)
        # Paint the items
        for item_c in item_channels:
            item = world_color_tensor[:, item_c]
            item_alpha = tensor_repr[:, item_c]
            image_repr = _draw_over(image_repr, item, item_alpha)
        # Finally paint the agent
        agent = world_color_tensor[:, c_name_to_idx["agent"]]
        agent_alpha = tensor_repr[:, c_name_to_idx["agent"]]
        image_repr = _draw_over(image_repr, agent, agent_alpha)

        return image_repr