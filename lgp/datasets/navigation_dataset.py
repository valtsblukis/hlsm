from abc import abstractmethod
import random
import torch
from typing import List, Dict, Union

from lgp.abcd.dataset import ExtensibleDataset
from lgp.rollout.rollout_data import load_rollout_from_path


class NavigationDataset(ExtensibleDataset):

    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths

    @abstractmethod
    def __getitem__(self, item):
        example = load_rollout_from_path(self.chunk_paths[item])
        example = self._process_example(example)
        return example

    @abstractmethod
    def __len__(self):
        return len(self.chunk_paths)

    def _padded_roll(self, inp, sy, sx):
        b, c, h, w = inp.shape
        canvas = torch.zeros((b, c, h*2, w*2), dtype=inp.dtype, device=inp.device)
        canvas[:, :, h//2:3*h//2, w//2:3*w//2] = inp
        canvas = torch.roll(canvas, shifts=(sy, sx), dims=(2, 3))
        outp = canvas[:, :, h//2:3*h//2, w//2:3*w//2]
        return outp

    def _process_example(self, example):
        f2d = example["features_2d"]
        simg = example["state_image"]
        b, c, h, w = f2d.shape

        # Compute the relative position of agent to the center of the image
        cy = h // 2
        cx = w // 2
        ay = example["current_pos"][0]
        ax = example["current_pos"][1]
        sy = int(cy - ay)
        sx = int(cx - ax)

        # Create a shifted state image centered around the agent
        s_img = self._padded_roll(simg, sy, sx)

        # Create a shifted feature image centered around the agent
        f2d_r = self._padded_roll(f2d, sy, sx)

        # Create a shifted action hl argument mask with 2 channels:
        #  1st channel indicates where the argument is and isn't
        #  2nd channel is the mean argument height in meters relative to the CAMERA_HEIGHT
        argmask_2d = example["subgoal"].get_spatial_arg_2d_features()
        argmask_2d_r = self._padded_roll(argmask_2d, sy, sx)

        # Create a goal image shifted around the agent, and also compute the relative goal position
        gy = example["nav_goal_pos"][0]
        gx = example["nav_goal_pos"][1]
        rgy = int(gy - ay + cy)
        rgx = int(gx - ax + cx)
        g_yaw_bin = example["goal_yaw_bin"]

        # If goal is out-of-bounds, discard the example
        if not (0 <= rgy < h and 0 <= rgx < w):
            example_out = None
            return example_out

        # print("Yaw bin: ", g_yaw_bin)
        #nav_goal_rel = torch.tensor([[rgy, rgx, 0]], device=f2d.device, dtype=torch.long)
        g_pitch_rad, g_yaw_rad = example["nav_goal_rot"]

        # Create tensor representations of the hl sem actions
        subgoal_t = example["subgoal"].to_tensor()

        g_img = torch.zeros((b, 4, h, w), dtype=f2d.dtype, device=f2d.device)
        g_pitch_img = torch.zeros((b, 4, h, w), dtype=f2d.dtype, device=f2d.device)

        ####################
        # Data Augmentation
        ####################

        # And flip with probability p=0.5
        flipfn = lambda m: m.flip(dims=[3])
        rotfn = lambda m: m.rot90(x, dims=[2, 3])

        # First flip and rotate the inputs and calculate yaw with respect to these augmented inputs
        doflip = random.randint(0, 1)
        if doflip:
            s_img, f2d_r, argmask_2d_r, g_pitch_img = map(flipfn, [s_img, f2d_r, argmask_2d_r, g_pitch_img])
            g_yaw_bin = (-g_yaw_bin) % 4
        x = random.randint(0, 3)
        if x > 0:
            s_img, f2d_r, argmask_2d_r, g_pitch_img = map(rotfn, [s_img, f2d_r, argmask_2d_r, g_pitch_img])
            g_yaw_bin = (g_yaw_bin + x) % 4

        # Goal and pitch image needs special treatment of the channel axis which encodes the yaw
        g_img[:, g_yaw_bin, rgy, rgx] = 1.0
        g_pitch_img[:, g_yaw_bin, rgy, rgx] = g_pitch_rad
        if doflip:
            g_img, g_pitch_img = map(flipfn, [g_img, g_pitch_img])
        if x > 0:
            g_img, g_pitch_img = map(rotfn, [g_img, g_pitch_img])

        example_out = {
            "state_image_rel": s_img,
            "features_2d_rel": f2d_r,
            "goal_img_rel": g_img,
            "goal_pitch_img_rel": g_pitch_img,
            "subgoal_arg_rel": argmask_2d_r,
            "subgoal_tensor": subgoal_t,
        }
        return example_out

    # Inherited from lgp.abcd.dataset.ExtensibleDataset
    def collate_fn(self, list_of_examples: Union[List[Dict], List[List[Dict]]]) -> Dict:
        list_of_examples = [l for l in list_of_examples if l is not None]
        print(len(list_of_examples))

        state_images = [l["state_image_rel"] for l in list_of_examples]
        features_2d = [l["features_2d_rel"] for l in list_of_examples]
        subgoals = [l["subgoal_tensor"] for l in list_of_examples]
        subgoal_args = [l["subgoal_arg_rel"] for l in list_of_examples]
        nav_goal_images = [l["goal_img_rel"] for l in list_of_examples]
        nav_goal_pitch_images = [l["goal_pitch_img_rel"] for l in list_of_examples]

        state_images = torch.cat(state_images, dim=0)
        features_2d = torch.cat(features_2d, dim=0)
        subgoals = torch.cat(subgoals, dim=0)
        subgoal_args = torch.cat(subgoal_args, dim=0)
        nav_goal_images = torch.cat(nav_goal_images, dim=0)
        nav_goal_pitch_images = torch.cat(nav_goal_pitch_images, dim=0)

        out = {
            "state_images": state_images,
            "features_2d": features_2d,
            "subgoals": subgoals,
            "subgoal_args": subgoal_args,
            "nav_goal_images": nav_goal_images,
            "nav_goal_pitch_images": nav_goal_pitch_images
        }
        return out
