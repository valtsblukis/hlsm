from typing import Dict, Union

import copy
import torch
import math
from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_observation import AlfredObservation
from lgp.env.alfred.tasks import AlfredTask
from lgp.env.alfred.alfred_action import AlfredAction, ACTION_TYPES
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.models.alfred.handcoded_skills.rotate_to_yaw import RotateToYawSkill
from lgp.models.alfred.handcoded_skills.tilt_to_pitch import TiltToPitchSkill
from lgp.models.alfred.handcoded_skills.go_to import GoToSkill

from lgp.ops.spatial_ops import unravel_spatial_arg, ravel_spatial_arg

import lgp.env.blockworld.config as config


PREDICT_EVERY_N = 50


class GoForManualSkill(Skill):
    def __init__(self):
        super().__init__()
        self.goto_skill = GoToSkill()
        self.rotate_to_yaw = RotateToYawSkill()
        self.tilt_to_pitch = TiltToPitchSkill()

        # make sure all of these are in _reset
        self.subgoal = None
        self.goto_done = False
        self.rotate_to_yaw_done = False
        self.tilt_to_pitch_done = False
        self.act_count = 0

        self.trace = {}

    def _reset(self):
        self.subgoal = None
        self.goto_done = False
        self.rotate_to_yaw_done = False
        self.tilt_to_pitch_done = False
        self.act_count = 0
        self.trace = {}

    def _construct_cost_function(self, state_repr : AlfredSpatialStateRepr):
        b, c, w, l, h = state_repr.data.data.shape
        spatial_arg = self.subgoal.build_spatial_arg_proposal(state_repr)

        max_response = spatial_arg.max().item()
        if max_response < 1e-10:
            print(f"WHOOPS! OBJECT {self.subgoal.arg_str()} NOT FOUND!")

        goal_coord_flat = torch.argmax(spatial_arg.view([b, -1]), dim=1)
        x, y, z = unravel_spatial_arg(goal_coord_flat, w, l, h) # Unravel goal_coord_flat
        goal_coord_vx = torch.stack([x, y, z], dim=1)
        goal_coord_m = (goal_coord_vx * state_repr.data.voxel_size) + state_repr.data.origin
        coords_xy = state_repr.data.get_centroid_coord_grid()[:, 0:2, :, :, 0] - goal_coord_m[:, 0:2, None, None]
        cx = coords_xy[:, 0]
        cy = coords_xy[:, 1]

        NOMINAL_INTERACT_DIST = 0.8
        NOMINAL_INTERACT_ANGLE = 0.2 # was 0.2

        def _cost_fn_a(cx, cy, intract_dist, interact_angle):
            distance_cost = -torch.exp(-(torch.sqrt(cx**2 + cy**2) - intract_dist) ** 2)
            direction_cost_a = torch.exp(-(torch.sin(torch.atan2(cy, cx)) ** 2) / interact_angle)
            direction_cost_b = torch.exp(-(torch.cos(torch.atan2(cy, cx)) ** 2) / interact_angle)
            full_pos_cost = distance_cost * (direction_cost_a + direction_cost_b)
            full_pos_reward = -full_pos_cost
            # Squash to 0-1 range
            full_pos_reward = full_pos_reward - torch.min(full_pos_reward)
            full_pos_reward = full_pos_reward / torch.max(full_pos_reward)
            return full_pos_reward

        def _cost_fn_b(cx, cy, interact_dist, interact_angle):
            T = 0.3
            interact_dist = 0.6
            x_match = (interact_dist - T < cx.abs()).logical_and(interact_dist + T > cx.abs()).logical_and(T > cy.abs()).logical_and(-T < cy.abs())
            y_match = (interact_dist - T < cy.abs()).logical_and(interact_dist + T > cy.abs()).logical_and(T > cx.abs()).logical_and(-T < cx.abs())
            match = x_match.logical_or(y_match)
            full_pos_reward = match.float()
            return full_pos_reward

        _cost_fn = _cost_fn_a

        full_pos_reward = _cost_fn(cx, cy, NOMINAL_INTERACT_DIST, NOMINAL_INTERACT_ANGLE)
        return full_pos_reward[:, None, :, :]

    def get_trace(self, device="cpu") -> Dict:
        trace = {
            "goto": self.goto_skill.get_trace(device),
            "rotatetoyaw": self.rotate_to_yaw.get_trace(device),
        }
        if "goal" in self.trace:
            trace["goal"] = self.trace["goal"]
        return trace

    def clear_trace(self):
        self.goto_skill.clear_trace()
        self.rotate_to_yaw.clear_trace()
        self.trace = {}

    def has_failed(self) -> bool:
        return False

    def set_goal(self, subgoal : AlfredSubgoal, remember_past_failures=False):
        self._reset()
        self.subgoal = subgoal
        #self.trace["goal"] = self.goal.get_argument_mask()argument_mask.data

    def act(self, state_repr: AlfredSpatialStateRepr) -> AlfredAction:
        self.act_count += 1
        # Haven't yet gone to the goal position
        if not self.goto_done:
            if self.act_count % PREDICT_EVERY_N == 1:
                rewardmap = self._construct_cost_function(state_repr)
                self.goto_skill.set_goal(rewardmap)
            action = self.goto_skill.act(state_repr)
            if action.is_stop():
                self.goto_done = True
            else:
                return action
        # Have gone to the goal position, rotate to face the goal
        if not self.rotate_to_yaw_done:
            a_x_vx, a_y_vx, a_z_vx = state_repr.get_pos_xyz_vx()
            g_x_vx, g_y_vx, g_z_vx = self.subgoal.get_argmax_spatial_arg_pos_xyz_vx()
            target_yaw = math.atan2(g_y_vx - a_y_vx, g_x_vx - a_x_vx + 1e-10)
            self.rotate_to_yaw.set_goal(target_yaw)
            action = self.rotate_to_yaw.act(state_repr)
            if action.is_stop():
                self.rotate_to_yaw_done = True
            else:
                return action

        if not self.tilt_to_pitch_done:
            a_x_vx, a_y_vx, a_z_vx = state_repr.get_pos_xyz_vx()
            g_x_vx, g_y_vx, g_z_vx = self.subgoal.get_argmax_spatial_arg_pos_xyz_vx()
            a_z_vx = -a_z_vx
            delta_z = (g_z_vx - a_z_vx).item()
            delta_x = (g_x_vx - a_x_vx).item()
            delta_y = (g_y_vx - a_y_vx).item()
            delta_xy = math.sqrt(delta_x ** 2 + delta_y ** 2)
            target_pitch = math.atan2(-delta_z, delta_xy + 1e-10)
            self.tilt_to_pitch.set_goal(target_pitch)
            action = self.tilt_to_pitch.act(state_repr)
            if action.is_stop():
                self.tilt_to_pitch_done = True
            else:
                return action

        # Finally, if finished going to the position and rotating, report "STOP
        return AlfredAction(action_type="Stop", argument_mask=None)