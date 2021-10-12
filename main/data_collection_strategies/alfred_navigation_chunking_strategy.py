from typing import List, Dict

import copy
import math

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
#from lgp.models.alfred.hlsm.hlsm_action_repr import HlsmActionRepr
from lgp.models.alfred.voxel_grid import VoxelGrid

# TODO: HL->Subgoal check


class AlfredNavigationPreproc:
    def __init__(self):
        ...

    def process_sample(self, sample):
        pass

    def process(self, rollout):
        return rollout


class NavToGoalChunkingStrategy:
    def __init__(self):
        ...

    #def _interaction_object_observed(self, state_repr: AlfredSpatialStateRepr, action_repr: HlsmActionRepr):
    #    observability_action_intersection = state_repr.obs_mask.data * action_repr.spatial_argument.data
    #    observed = observability_action_intersection.sum().detach().item() > 0.5
    #    return observed

    def is_sequence_terminal(self, action: AlfredAction):
        return action.action_type in AlfredAction.get_interact_action_list() or action.is_stop()

    def include_chunk(self, action: AlfredAction):
        return not action.is_stop()

    def ll_to_hl(self, samples: List[Dict], start_idx : int):
        rollout_out = []

        last_action = samples[-1]["action"]
        last_observation = samples[-1]["observation"]

        # 1. Construct a sequence of 2D feature inputs containing:
        #   - Obstacle Map
        #   - Receptacle Map
        #   - Pickable Map
        #   - ... etc
        nav_features_2d_all = {t : samples[t]["state_repr"].get_nav_features_2d() for t in range(start_idx, len(samples), 1)}

        for t in range(start_idx, len(samples), 1):

            features_2d = nav_features_2d_all[t]
            features_2d_final = nav_features_2d_all[len(samples) - 1]

            # Where the agent is right now
            x_vx, y_vx, z_vx = samples[t]["state_repr"].get_pos_xyz_vx()
            x_vx, y_vx, z_vx = x_vx.item(), y_vx.item(), z_vx.item()
            pitch, yaw = samples[t]["state_repr"].get_camera_pitch_yaw()

            # Look ahead where the agent went next:
            for tk in range(t, len(samples), 1):
                x_vx_next, y_vx_next, z_vx_next = samples[tk]["state_repr"].get_pos_xyz_vx()
                x_vx_next, y_vx_next, z_vx_next = x_vx_next.item(), y_vx_next.item(), z_vx_next.item()
                if x_vx_next != x_vx or y_vx_next != y_vx:
                    break

            future_positions = []
            for tj in range(t+1, len(samples), 1):
                x_vx_fut, y_vx_fut, z_vx_fut = samples[tj]["state_repr"].get_pos_xyz_vx()
                x_vx_fut, y_vx_fut, z_vx_fut = x_vx_fut.item(), y_vx_fut.item(), z_vx_fut.item()
                future_positions.append((x_vx_fut, y_vx_fut, z_vx_fut))

            # Which bin does the rotation fall into
            yaw_pos = (yaw + 1e-3) % (math.pi * 2)
            yaw_bin = yaw_pos / (math.pi / 2)

            # Where the agent ended up
            goal_x_vx, goal_y_vx, goal_z_vx = samples[-1]["state_repr"].get_pos_xyz_vx()
            goal_x_vx, goal_y_vx, goal_z_vx = goal_x_vx.item(), goal_y_vx.item(), goal_z_vx.item()
            goal_pitch, goal_yaw = samples[-1]["state_repr"].get_camera_pitch_yaw()

            # Which bin does the goal rotation fall into
            goal_yaw_pos = (goal_yaw + 1e-3) % (math.pi * 2)
            goal_yaw_bin = int(goal_yaw_pos / (math.pi / 2))

            # What the agent was doing spatially
            if last_action.is_stop():
                subgoal = AlfredSubgoal.from_type_str_and_arg_id("Stop", -1)
            else:
                subgoal = AlfredSubgoal.from_action_and_observation(last_action, last_observation)

            # State image for later visualizing
            state_image = samples[t]["state_repr"].represent_as_image(topdown2d=True)

            # Observation for learning segmentation and monocular depth
            observation = copy.deepcopy(samples[t]["observation"])
            # Just delete the AI2Thor event because that takes up a lot of space!
            del observation.privileged_info._world_state
            # Convert to a more compact integer segmentation representation
            observation.compress()

            sample_out = {
                "state_image": state_image.to("cpu"),
                "observation": observation.to("cpu"),
                "features_2d": features_2d.to("cpu"),
                "features_2d_final": features_2d_final.to("cpu"),
                "subgoal": subgoal.to("cpu"),
                "current_pos": (x_vx, y_vx, z_vx),
                "all_future_pos": future_positions,
                "next_pos": (x_vx_next, y_vx_next, z_vx_next),
                "current_rot": (pitch, yaw),
                "current_yaw_bin": yaw_bin,
                "nav_goal_pos": (goal_x_vx, goal_y_vx, goal_z_vx),
                "nav_goal_rot": (goal_pitch, goal_yaw),
                "goal_yaw_bin": goal_yaw_bin
            }
            rollout_out.append(sample_out)
        return rollout_out
