from typing import Dict

import torch
import torch.nn.functional as F

from typing import Dict

from lgp.abcd.skill import Skill

import lgp.env.alfred.segmentation_definitions as segdef

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.models.alfred.handcoded_skills.go_to import GoToSkill

import lgp.paths

MAX_NAV_COUNT = 40
SIMPLE_EXPLORE_BASELINE = False

SAMPLE_FROM = "ground_edge"


FOUND_THRESHOLD = 0.3


class ExploreSkill(Skill):
    def __init__(self):
        super().__init__()
        self.goto_skill = GoToSkill()
        self._reset()

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.hl_action = None
        self.pre_rotation_finished = False
        self.navigate_finished = False
        self.post_rotation_finished = False
        self.found = False
        self.rotation_count = 0
        self.count = 0
        self.trace = {}

        self.rewardmap = None

    def _build_floor_edge_mask(self, state_repr):
        floor_ids = [segdef.object_string_to_intid(s) for s in ["Floor"]]
        wall_ids = [segdef.object_string_to_intid(s) for s in ["Wall", "StandardWallSize", "Door"]]
        map_floor = state_repr.data.data[:, floor_ids].max(1).values.max(3).values.float()
        map_wall = state_repr.data.data[:, wall_ids].max(1).values.max(3).values.float()
        map_occupied = state_repr.data.occupancy.max(1).values.max(3).values.float()
        map_free = 1 - map_occupied

        map_observed = state_repr.obs_mask.data.max(1).values.max(3).values.float()
        map_unobserved = 1 - map_observed

        kern1 = torch.ones((1, 1, 3, 3), device=map_free.device)
        expanded_unobserved = (F.conv2d(map_unobserved[None, :, :, :], kern1, padding=1)[0] > 0.5).float()
        expanded_walls = (F.conv2d(map_wall[None, :, :, :], kern1, padding=1)[0] > 0.5).float()
        not_expanded_walls = 1 - expanded_walls

        unobserved_floor_boundary = expanded_unobserved * map_floor * not_expanded_walls
        return unobserved_floor_boundary

    def _construct_cost_function(self, state_repr : AlfredSpatialStateRepr):
        b, c, w, l, h = state_repr.data.data.shape

        ground_2d = (state_repr.data.occupancy[:, :, :, :, 0:4].sum(4) == 1).float()

        filt_pos_prob = self._build_floor_edge_mask(state_repr)
        # Proposals on ground if ground_edge doesn't exist
        if filt_pos_prob.sum() < 0.5:
            filt_pos_prob = ground_2d + 1e-10

        filt_pos_prob = filt_pos_prob / filt_pos_prob.sum()

        # TODO: Figure out why sometimes there are 3 dimensions, sometimes 4
        if len(filt_pos_prob.shape) == 3:
            filt_pos_prob = filt_pos_prob[None, :, :, :]

        idx = torch.distributions.categorical.Categorical(probs=filt_pos_prob.view([-1])).sample().item()
        x = idx // l
        y = idx % l
        self.rewardmap = torch.zeros_like(filt_pos_prob)
        self.rewardmap[:, :, x, y] = 1.0

        return self.rewardmap

    def get_trace(self, device="cpu") -> Dict:
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def has_failed(self) -> bool:
        return False

    def _is_found(self, state_repr):
        weighted_layers = state_repr.data.data * self.hl_action.argument_vector[:, 1:, None, None, None]
        response = weighted_layers.sum().item()
        if response >= FOUND_THRESHOLD:
            return True
        else:
            return False

    def set_goal(self, hl_action : AlfredSubgoal):
        self._reset()
        self.hl_action = hl_action

    def act(self, state_repr: AlfredSpatialStateRepr) -> AlfredAction:
        if self.rewardmap is None:
            self.rewardmap = self._construct_cost_function(state_repr)
            self.goto_skill.set_goal(self.rewardmap)

        if self.count > MAX_NAV_COUNT:
            self.navigate_finished = True
            self.pre_rotation_finished = True

        # Baseline with 4 in-place rotations
        if SIMPLE_EXPLORE_BASELINE:
            self.pre_rotation_finished = True
            self.navigate_finished = True

        # Always look up and rotate around, even if we think that the object is found.
        # This costs very little, but can significantly improve the map representation between
        # failed navigation retries.
        if not self.pre_rotation_finished:
            action : AlfredAction = AlfredAction(action_type="RotateLeft", argument_mask=None)
            self.rotation_count += 1
            #if self.rotation_count == 1:
            #    action : AlfredAction = AlfredAction(action_type="LookUp", argument_mask=None)
            if self.rotation_count == 3:
                #action: AlfredAction = AlfredAction(action_type="LookDown", argument_mask=None)
                self.rotation_count = 0
                self.pre_rotation_finished = True

        # TODO: Re-check order of the next two if and elif.
        elif self._is_found(state_repr):
            self.navigate_finished = True
            self.pre_rotation_finished = True
            self.post_rotation_finished = True
            action = AlfredAction(action_type="Stop", argument_mask=None)

        elif not self.navigate_finished:
            # Sample a non-stop action, unless the object is found or time limit exceeded
            count = 0
            while True:
                if self.rewardmap is None:
                    self.rewardmap = self._construct_cost_function(state_repr)
                    self.goto_skill.set_goal(self.rewardmap)
                action : AlfredAction = self.goto_skill.act(state_repr)
                if action.is_stop():
                    self.rewardmap = None
                    count += 1
                    if count > 2:
                        self.navigate_finished = True
                        break
                else:
                    break

        elif not self.post_rotation_finished:
            # First do 4x RotateLeft to look around
            action : AlfredAction = AlfredAction(action_type="RotateLeft", argument_mask=None)
            self.rotation_count += 1

            # Look up before rotating
            if self.rotation_count == 1:
                action : AlfredAction = AlfredAction(action_type="LookUp", argument_mask=None)

            # Look back down after rotating
            if self.rotation_count == 5:
                action: AlfredAction = AlfredAction(action_type="LookDown", argument_mask=None)
                self.rotation_count = 0
                self.post_rotation_finished = True

        else:
            action = AlfredAction(action_type="Stop", argument_mask=None)

        self.count += 1
        return action