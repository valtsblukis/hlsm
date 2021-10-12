from typing import Dict

import torch
import math
from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.models.alfred.handcoded_skills.rotate_to_yaw import RotateToYawSkill
from lgp.models.alfred.handcoded_skills.tilt_to_pitch import TiltToPitchSkill
from lgp.models.alfred.handcoded_skills.go_to import GoToSkill
from lgp.models.alfred.hlsm.hlsm_navigation_model import HlsmNavigationModel

from lgp.ops.spatial_ops import unravel_spatial_arg
import lgp.paths

from lgp.flags import GLOBAL_VIZ

PREDICT_EVERY_N = 50

# Instead of predicting a yaw distribution to sample from, just turn towards the argmax action argument
LEGACY_YAW = False


class TriedPosYawGrid:

    def __init__(self, h, w):
        self.grid = torch.ones((1, 4, h, w)) # B x yaw x y x x

    def mark_attempt(self, y, x, ry, rx, yaw, pitch):
        yaw_id = int(((yaw % (2 * math.pi)) + 1e-3) / (math.pi / 2))
        # Mark intended goal as attempted
        self.grid[0, yaw_id, y, x] = 0.0
        # Mark actually reached goal as attempted too
        self.grid[0, yaw_id, ry, rx] = 0.0

    def get_pos_mask(self, device="cpu"):
        return self.grid.max(dim=1, keepdim=True).values.to(device)

    def get_yaw_mask(self, x, y, device="cpu"):
        return self.grid[:, :, x, y].to(device)


class GoForSkill(Skill):
    def __init__(self):
        super().__init__()
        self.goto_skill = GoToSkill()
        self.rotate_to_yaw = RotateToYawSkill()
        self.tilt_to_pitch = TiltToPitchSkill()

        self.navigation_model = HlsmNavigationModel()
        navsd = torch.load(lgp.paths.get_navigation_model_path())
        self.navigation_model.load_state_dict(navsd)

        # make sure all of these are in _reset
        self.subgoal = None
        self.goto_done = False
        self.rotate_yaw_done = False
        self.rotate_pitch_done = False
        self.act_count = 0

        self.rewardmap = None
        self.yawmap = None # B x 4 x H x W map that includes a yaw channel
        self.pitchmap = None
        self.target_yaw = None
        self.target_pitch = None
        self.tried_grid = None
        self.goal_pos = None

        self.trace = {}

    def start_new_rollout(self):
        self._reset()
        self.rotate_to_yaw.start_new_rollout()
        self.tilt_to_pitch.start_new_rollout()
        self.goto_skill.start_new_rollout()

    def _reset(self, remember_past_failures=False):
        self.subgoal = None
        self.goto_done = False
        self.rotate_yaw_done = False
        self.rotate_pitch_done = False
        self.act_count = 0

        self.rewardmap = None
        self.yawmap = None
        self.pitchmap = None
        self.target_yaw = None
        self.target_pitch = None
        self.goal_pos = None

        if not remember_past_failures:
            #print("GOFOR SKILL: CLEARING ATTEMPT HISTORY !!!")
            self.tried_grid = None
        else:
            #print("GOFOR SKILL: KEEPING ATTEMPT HISTORY !!!")
            pass # Keep the "tried grid" from the previous attempt

        self.trace = {}

    def _construct_cost_function(self, state_repr: AlfredSpatialStateRepr):
        features_2d_centered = state_repr.get_nav_features_2d(center_around_agent=True)

        # TOOD: Check the values are correct here
        act_arg_2d_features = self.subgoal.get_spatial_arg_2d_features()

        #spatial_arg = spatial_arg.max(dim=4).values
        spatial_action_arg_features_centered = state_repr.center_2d_map_around_agent(act_arg_2d_features)
        subgoal_tensor = self.subgoal.to_tensor(device=features_2d_centered.device)

        self.navigation_model = self.navigation_model.to(features_2d_centered.device)

        pos_pred_log_distr, yaw_pred_log_distr, pitch_prediction = self.navigation_model.forward_model(
            features_2d_centered, spatial_action_arg_features_centered, subgoal_tensor)

        pos_pred_distr = torch.exp(pos_pred_log_distr)  # 1x1xHxW map of P(x,y) position probabilities
        yaw_pred_distr = torch.exp(yaw_pred_log_distr)  # 1x4xHxW map of P(yaw | x,y) yaw probabilities conditioned on position
        # pitch_prediction is a 1x4xHxW map of E(pitch | yaw, x, y) # TODO: Consider binning this too and allow sampling a pitch
        pitch_prediction = pitch_prediction.clamp(-math.pi / 2 + math.radians(0.15), math.pi / 2 - math.radians(0.15))

        # Shift all the maps back to global frame
        pos_pred_distr = state_repr.center_2d_map_around_agent(pos_pred_distr, inverse=True)
        yaw_pred_distr = state_repr.center_2d_map_around_agent(yaw_pred_distr, inverse=True)
        pitch_prediction = state_repr.center_2d_map_around_agent(pitch_prediction, inverse=True)

        # Sample an x,y coordinate to go to
        SAMPLE = True
        if SAMPLE:
            b, c, h, w = pos_pred_distr.shape

            # Mask out (x, y, yaw) options that have already been tried
            if self.tried_grid is None:
                self.tried_grid = TriedPosYawGrid(h, w)
            untried_mask = self.tried_grid.get_pos_mask(device=pos_pred_distr.device)

            # Only allow sampling goals in free and observed space along positions that haven't been tried
            sampling_distr = pos_pred_distr * (1 - state_repr.get_obstacle_map_2d()) * state_repr.get_observability_map_2d(floor_only=True) * untried_mask
            sampling_distr = sampling_distr + 1e-20
            sampling_distr = sampling_distr / (sampling_distr.sum())

            # Sample an x,y,z
            posxy = torch.distributions.Categorical(probs=sampling_distr.reshape([b, -1])).sample_n(1)
            pos_y = (posxy // w).item()
            pos_x = (posxy % w).item()
            full_pos_reward = torch.zeros_like(pos_pred_distr)
            full_pos_reward[0, 0, pos_y, pos_x] = 1.0

            print(f"GO FOR SAMPLED GOAL: {pos_x}, {pos_y}")

            # Give partial credit for getting near the goal position
            partial_credit_kernel = torch.tensor(
                [[0.2, 0.4, 0.6, 0.4, 0.2],
                 [0.4, 0.6, 0.8, 0.6, 0.4],
                 [0.6, 0.8, 1.0, 0.8, 0.6],
                 [0.4, 0.6, 0.8, 0.6, 0.4],
                 [0.2, 0.4, 0.6, 0.4, 0.2]],
                device=full_pos_reward.device)[None, None, :, :]

            full_pos_reward = torch.conv2d(full_pos_reward, partial_credit_kernel, stride=1, padding=int(partial_credit_kernel.shape[2] / 2))

        # Compute probability over positions and set that as the goal map.
        else:
            full_pos_reward = pos_pred_distr.sum(dim=1, keepdim=True)
            pos_y, pos_x = 0, 0

        viz = GLOBAL_VIZ
        if viz:
            from lgp.utils.viz import show_image

            device = features_2d_centered.device
            s2d = state_repr.represent_as_image(topdown2d=True)
            colors = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]], device=device)
            colors4 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], device=device)
            f2d_colors = (features_2d_centered[:, :, None, :, :] * colors[None, :, :, None, None]).sum(dim=1) / 3

            s2d_np = s2d[0].permute((1, 2, 0)).detach().cpu().numpy()
            f2d_color_np = f2d_colors[0].permute((1, 2, 0)).detach().cpu().numpy()

            pos_pred_distr_viz = pos_pred_distr / (pos_pred_distr.max())
            # B x 4 x 3 x H x W
            pos_pred_distr_viz_np = pos_pred_distr_viz[0].permute((1, 2, 0)).detach().cpu().numpy()

            yaw_pred_distr_viz = yaw_pred_distr / (yaw_pred_distr.max())
            yaw_pred_distr_viz = (yaw_pred_distr_viz[:, :, None, :, :] * colors4[None, :, :, None, None]).sum(dim=1)
            yaw_pred_distr_viz_np = yaw_pred_distr_viz[0].permute((1, 2, 0)).detach().cpu().numpy()

            action_np = act_arg_2d_features[0, 0:1, :, :].permute((1, 2, 0)).detach().cpu().numpy()
            g_img_np = full_pos_reward[0].permute((1, 2, 0)).detach().cpu().numpy()

            s2d_and_pos_np = s2d_np * 0.3 + pos_pred_distr_viz_np * 0.7
            s2d_and_yaw_np = s2d_np * 0.3 + yaw_pred_distr_viz_np * 0.7
            s2d_and_act_np = s2d_np * 0.3 + action_np * 0.7
            s2d_and_g_np = s2d_np * 0.3 + g_img_np * 0.7

            show_image(s2d_np, "State", scale=4, waitkey=1)
            show_image(f2d_color_np, "Features", scale=4, waitkey=1)
            show_image(s2d_and_pos_np, "State + Goal pos prob", scale=4, waitkey=1)
            show_image(s2d_and_yaw_np, "State + Yaw prob", scale=4, waitkey=1)
            show_image(s2d_and_act_np, "State + Action arg", scale=4, waitkey=1)
            show_image(s2d_and_g_np, "State + Goal pos", scale=4, waitkey=1)

        return full_pos_reward, yaw_pred_distr, pitch_prediction, (pos_y, pos_x)

    def _construct_cost_function_manual(self, state_repr : AlfredSpatialStateRepr):
        b, c, w, l, h = state_repr.data.data.shape
        spatial_arg = self.subgoal.get_argument_mask()#state_repr)

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

        _cost_fn = _cost_fn_a

        full_pos_reward = _cost_fn(cx, cy, NOMINAL_INTERACT_DIST, NOMINAL_INTERACT_ANGLE)
        return full_pos_reward[:, None, :, :]

    def _sample_rotation_goal(self, state_repr):
        a_x_vx, a_y_vx, a_z_vx = state_repr.get_pos_xyz_vx()

        # If the agent is out-of-bounds, snap to the closest within-bounds position
        _, _, h, w = self.yawmap.shape
        xpos = min(max(int(a_x_vx), 0), h - 1)
        ypos = min(max(int(a_y_vx), 0), h - 1)

        # Only sample from yaw angles that haven't been attempted at this position before
        yaw_mask = self.tried_grid.get_yaw_mask(xpos, ypos, device=state_repr.data.data.device)

        # Sample a yaw angle to turn to
        yawdistr = self.yawmap[:, :, xpos, ypos] + 1e-3
        yawdistr = yawdistr * yaw_mask + 1e-6
        yawdistr = yawdistr / yawdistr.sum()
        yaw_id = torch.distributions.Categorical(probs=yawdistr).sample_n(1)
        target_yaw = yaw_id * (math.pi / 2)

        # Grab the pitch angle to tilt the head to - this is computed using regression and is not distributional
        target_pitch = self.pitchmap[:, :, xpos, ypos]

        return target_yaw, target_pitch

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

    def set_goal(self, subgoal : AlfredSubgoal, remember_past_failures : bool = False):
        self._reset(remember_past_failures)
        self.subgoal = subgoal

    def remember_final_pose(self, state_repr):
        a_x_vx, a_y_vx, a_z_vx = state_repr.get_pos_xyz_vx()
        b, c, w, l, h = state_repr.data.data.shape
        reached_ypos = min(max(int(a_x_vx), 0), w - 1)
        reached_xpos = min(max(int(a_y_vx), 0), l - 1)

        # Mark based on the sampled goal, not the reached goal.
        # Sometimes a goal out of bounds is sampled, and we want to mark that down rather than penalizing the position
        # where the agent did reach even though it may not have intended to reach
        ypos, xpos = self.goal_pos
        _, _, yaw = state_repr.get_rpy()
        pitch = state_repr.get_camera_pitch_deg()
        self.tried_grid.mark_attempt(ypos, xpos, reached_ypos, reached_xpos, yaw, pitch)

    def has_failed(self) -> bool:
        return False

    def act(self, state_repr: AlfredSpatialStateRepr) -> AlfredAction:
        self.act_count += 1

        # Haven't yet gone to the goal position
        if not self.goto_done:

            # First time, and then every N times. Less frequent updates avoid oscillations.
            if self.act_count % PREDICT_EVERY_N == 1:
                self.rewardmap, self.yawmap, self.pitchmap, self.goal_pos = self._construct_cost_function(state_repr)
                self.goto_skill.set_goal(self.rewardmap)

            self.trace["goal"] = self.subgoal.get_argument_mask()
            # TODO: Right now we are re-setting the goal on the goto skill every time.
            # this results in a value iteration network run every time too.
            # should we run the value iteration network less frequently?
            action = self.goto_skill.act(state_repr)
            if action.is_stop():
                self.goto_done = True
                self.target_yaw, self.target_pitch = self._sample_rotation_goal(state_repr)
            else:
                return action

        # Have gone to the goal position, rotate to face the goal
        if not self.rotate_yaw_done:
            a_x_vx, a_y_vx, a_z_vx = state_repr.get_pos_xyz_vx()
            self.trace["goal"] = self.subgoal.get_argument_mask()

            g_x_vx, g_y_vx, g_z_vx = self.subgoal.get_argmax_spatial_arg_pos_xyz_vx()#state_repr)
            legacy_target_yaw = math.atan2(g_y_vx - a_y_vx, g_x_vx - a_x_vx + 1e-10)

            if LEGACY_YAW:
                self.rotate_to_yaw.set_goal(legacy_target_yaw)
            else:
                self.rotate_to_yaw.set_goal(self.target_yaw)

            action = self.rotate_to_yaw.act(state_repr)
            if action.is_stop():
                self.rotate_yaw_done = True
                self.tilt_to_pitch.set_goal(self.target_pitch)
            else:
                return action

        # Pitch according to the prediction
        if not self.rotate_pitch_done:
            action = self.tilt_to_pitch.act(state_repr)
            if action.is_stop():
                self.rotate_pitch_done = True
            else:
                return action

        # Remember the final position and yaw so that we can avoid sampling it again
        self.remember_final_pose(state_repr)

        # Finally, if finished going to the position and rotating, report "STOP
        return AlfredAction(action_type="Stop", argument_mask=None)