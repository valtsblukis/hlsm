import torch
import os
import math
import random
import imageio

import lgp.env.alfred.segmentation_definitions as segdef

import lgp.paths
DEFAULT_PATH = os.path.join(lgp.paths.ROOT_PATH, "data", "textures", "img")


class TextureBank():
    def __init__(self, path=DEFAULT_PATH):
        fnames = [os.path.join(path, f) for f in os.listdir(path)]
        self.texture_bank = {}
        for i, f in enumerate(fnames):
            print(f"Loading texture {i}/{len(fnames)}")
            self.texture_bank[f] = imageio.imread(f)
        print("ding")


#TEXTUREBANK = TextureBank()


def random_offset_maps(seg_image):
    b, c, h, w = seg_image.shape
    add_maps = torch.zeros((b, c, 3, h, w), device=seg_image.device, dtype=torch.float32)
    mul_maps = torch.ones((b, c, 3, h, w), device=seg_image.device, dtype=torch.float32)

    # Additive color offset
    if random.random() > 0.5:
        flat_add_offset_std = random.random() * 0.2
        #print(f"Additive color offset: {flat_add_offset_std}")
        flat_add_offset = torch.normal(torch.zeros(b, c, 3, 1, 1), flat_add_offset_std)
        add_maps = add_maps + flat_add_offset

    # Multiplicative color offset
    if random.random() > 0.5:
        flat_mul_offset_std = random.random() * 0.2
        #print(f"Multiplicative color offset: {flat_mul_offset_std}")
        flat_mul_offset = torch.normal(torch.ones(b, c, 3, 1, 1), flat_mul_offset_std)
        mul_maps = mul_maps * flat_mul_offset

    # Additive gaussian noise
    if random.random() > 0.5:
        noise_offset_std = random.random() * 0.05
        #print(f"Additive gaussian noise : {noise_offset_std}")
        add_maps = torch.normal(add_maps, noise_offset_std)

    return add_maps, mul_maps


def seg_region_augmentation(observation):
    # For each segmentation class:
    b, c, h, w = observation.semantic_image.shape
    additive_maps, multiplicative_maps = random_offset_maps(observation.semantic_image)

    # Apply a separate data augmentation procedure for every object in the environment
    for i in range(c):
        # Only modify colors of structural objects
        if segdef.object_intid_to_string(i) not in segdef.AUGMENTATION_OBJECTS:
            continue
        mask = observation.semantic_image[:, i:i+1]
        if mask.sum() > 0:
            img_add_map = mask * additive_maps[:, i]
            img_mul_map = mask * multiplicative_maps[:, i] + (1 - mask) * torch.ones_like(observation.rgb_image)
            observation.rgb_image = observation.rgb_image * img_mul_map
            observation.rgb_image = observation.rgb_image + img_add_map

    observation.rgb_image = torch.clamp(observation.rgb_image, 0, 1)


def flip_augmentation(observation):
    if random.random() > 0.5:
        observation.rgb_image = torch.flip(observation.rgb_image, dims=(3,))
        observation.semantic_image = torch.flip(observation.semantic_image, dims=(3,))
        observation.depth_image = torch.flip(observation.depth_image, dims=(3,))
