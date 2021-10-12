from typing import Dict

import math
import torch
import torch.nn.functional as F

from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction

from lgp.models.alfred.handcoded_skills.rotate_to_yaw import RotateToYawSkill
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr


from lgp.utils.viz import show_image
from lgp.flags import GLOBAL_VIZ

GRID_SIZE = 61

TERMINAL_OBSTACLE_REWARD = -0.9    # Reward for colliding with obstacles
TERMINAL_STOP_GOAL_REWARD = 1.0    # Reward for stopping at the goal position
TERMINAL_STOP_ALL_REWARD = 0.001   # Reward for stopping at a non-goal position.
                                   # This should be slightly positive, so that if the goal is unreachable, the agent stops
                                   # if it's negative, the agent will roam around forever because a negative reward later
                                   # is better than a negative reward now

# Reward for passing through space where ground hasn't been observed
# This should be bigger than TERMINAL_STOP_GOAL_REWARD * gamma
INTERMEDIATE_UNOBSERVED_REWARD = -0.02

S_FREE = 0
S_OBSTACLE = 1
S_GOAL = 2
S_UNSEEN = 3
S_ALL = 4


class ValueIterationNetwork3D():

    def __init__(self, rewardmap=None, occupancy_map=None, observability_map=None, extra_obstacle_map=None):
        self.rewardmap = rewardmap
        self.occupancy_map = occupancy_map
        self.observability_map = observability_map
        self.extra_obstacle_map = extra_obstacle_map
        self._idx_to_gridaction = {
            0: "UP",
            1: "RIGHT",
            2: "DOWN",
            3: "LEFT",
            4: "STOP"
        }
        self._gridaction_to_idx = {v: k for k, v in self._idx_to_gridaction.items()}

    def _spatial_remap(self, x):
        return F.interpolate(input=x, size=GRID_SIZE)

    def set_rewardmap(self, rewardmap):
        self.rewardmap = self._spatial_remap(rewardmap.float())

    def set_occupancy_map(self, occupancy_map):
        self.occupancy_map = self._spatial_remap(occupancy_map.float())

    def set_observability_map(self, observability_map):
        self.observability_map = self._spatial_remap(observability_map.float())

    def set_extra_obstacle_map(self, extra_obstacle_map):
        self.extra_obstacle_map = self._spatial_remap(extra_obstacle_map.float())

    def idx_to_gridaction(self, idx):
        return self._idx_to_gridaction[idx]

    def _init_state_image(self):
        # Compute a binary image of all positions that are guaranteed to be free of obstacles
        OBSTACLE_EXPANSION = 0
        batch_size = self.occupancy_map.shape[0]
        state_map = torch.zeros((batch_size, 5, GRID_SIZE, GRID_SIZE), device=self.occupancy_map.device)
        K = torch.ones([1, 1, 1 + 2 * OBSTACLE_EXPANSION, 1 + 2 * OBSTACLE_EXPANSION], device=self.occupancy_map.device)
        occupancy_map_expanded = F.conv2d(self.occupancy_map, K, padding=OBSTACLE_EXPANSION).clamp(0, 1)[0]

        obstacle_map = (occupancy_map_expanded + self.extra_obstacle_map).clamp(0, 1)
        safe_freespace_map = 1 - obstacle_map
        unobserved_map = 1 - self.observability_map

        goal_map = self.rewardmap
        # State:
        # 0 - freespace
        # 1 - obstacle
        # 2 - rewardstate
        # 3 - unobserved
        # 4 - one
        state_map[:, S_FREE] = safe_freespace_map
        state_map[:, S_OBSTACLE] = 1 - safe_freespace_map
        state_map[:, S_GOAL] = goal_map
        state_map[:, S_UNSEEN] = unobserved_map
        state_map[:, S_ALL] = 1
        return state_map

    def _init_transition_kernel(self):
        # Actions:
        # 0 - UP
        # 1 - RIGHT
        # 2 - DOWN
        # 3 - LEFT
        # 4 - OTHER

        # TRANSITION KERNEL
        A = 0.01
        B = 1 - 8 * A
        p_kernel = torch.ones((5, 1, 3, 3), device=self.occupancy_map.device) * A
        p_kernel[self._gridaction_to_idx["UP"], 0, 0, 1] = B
        p_kernel[self._gridaction_to_idx["RIGHT"], 0, 1, 2] = B
        p_kernel[self._gridaction_to_idx["DOWN"], 0, 2, 1] = B
        p_kernel[self._gridaction_to_idx["LEFT"], 0, 1, 0] = B
        p_kernel[self._gridaction_to_idx["STOP"], 0, 1, 1] = B
        return p_kernel

    def _init_intermediate_reward_kernel(self):
        intermediate_r_kernel = torch.zeros((5, 5, 1, 1), device=self.occupancy_map.device)
        intermediate_r_kernel[0:5, S_UNSEEN] = INTERMEDIATE_UNOBSERVED_REWARD
        return intermediate_r_kernel

    def _init_terminal_reward_kernel(self):
        terminal_r_kernel = torch.zeros((5, 5, 1, 1), device=self.occupancy_map.device)
        terminal_r_kernel[0:5, S_OBSTACLE] = TERMINAL_OBSTACLE_REWARD
        terminal_r_kernel[4, S_ALL] = TERMINAL_STOP_ALL_REWARD
        terminal_r_kernel[4, S_GOAL] = TERMINAL_STOP_GOAL_REWARD
        return terminal_r_kernel

    def _initialize(self):
        state_map = self._init_state_image()
        t_kernel = self._init_transition_kernel()
        intermediate_r_kernel = self._init_intermediate_reward_kernel()
        terminal_r_kernel = self._init_terminal_reward_kernel()
        return state_map, t_kernel, intermediate_r_kernel, terminal_r_kernel

    def vin_module_q(self, Q, state_image, T, iRK, tRK, gamma=0.99):
        """
        Perform one step of value iteration on a grid-MDP with:
         - HxW states aligned in a spatial grid, where each state has S semantic properties.
         - A actions with effects described by a 3x3 matrix of transition probabilities.

        Args:
            Q: Q-image of the expected sum of future rewards when taking one of A actions in each of HxW states. Shape: 1xAxHxW
            state_image: Tensor that for each state carries a one-hot vector of S state properties. Shape: 1xSxHxW
            T: MDP transition probabilities. Shape: 1xAx3x3
            iRK: Intermediate reward kernel that specifies for each action and each state property, how much that
                 state property contributes to the intermediate step reward. Shape: AxSx1x1
            tRK: Terminal reward kernel that specifies for each action and each state property, how much that
                 state property contributes to the terminal reward.
                 Any state with a non-zero terminal reward is treated as a terminal state. Shape: AxSx1x1
            gamma: MDP discount factor

        Returns:
            Updated Q-image of shape 1xSxHxW
        """
        # Compute state value assuming greedy policy
        V = torch.max(Q, dim=1, keepdim=True).values

        # Propagate values to neighboring states according to the transition probabilities
        Vn = gamma * torch.conv2d(V, T, stride=1, padding=int((T.shape[2]-1)/2))

        # Compute intermediate reward R(s,a) for each action in each state
        iR = torch.conv2d(state_image, iRK, stride=1, padding=0)

        # Compute terminal reward R(s,a) for each action in each state.
        # If this is non-zero for a given (s,a), then that (s,a) is assumed to be episode-terminating
        tR = torch.conv2d(state_image, tRK, stride=1, padding=0)
        is_terminal = (tR != 0).float()

        # Compute Q(s,a)
        # This is:
        #    V(next state) + R(s,a) for all non-terminal (s,a)
        #    R(s,a) for all terminal (s,a) that don't have a next state
        Q = (Vn + iR) * (1 - is_terminal) + tR * is_terminal
        return Q

    def compute_q_image(self):
        w = self.rewardmap.shape[2]
        # Reset iteration
        q_image = torch.zeros_like(self.rewardmap)
        state_image, t_kernel, intermediate_r_kernel, terminal_r_kernel = self._initialize()
        for i in range(w * 2):
            q_image = self.vin_module_q(q_image, state_image, t_kernel, intermediate_r_kernel, terminal_r_kernel)
        return q_image

    def remap_xy(self, x, y, voxelgrid_size):
        scale = GRID_SIZE / voxelgrid_size
        x = int(x * scale)
        y = int(y * scale)
        return x, y


class GoToSkill(Skill):
    def __init__(self):
        super().__init__()
        self.rotate_to_yaw_skill = RotateToYawSkill()
        self.vin = ValueIterationNetwork3D()

        self.count = 0
        self.trace = {}
        self.clear_trace()
        self.prev_pos = (0, 0)
        self.prev_act = None
        self.tried_and_failed_actions = []
        self.bumper_obstacle_map = None

    def _reset(self):
        self.count = 0
        self.prev_pos = (0, 0)
        self.prev_act = None
        self.tried_and_failed_actions = []
        self.bumper_obstacle_map = None
        self.clear_trace()

    def start_new_rollout(self):
        self._reset()

    def set_goal(self, rewardmap):
        self.vin.set_rewardmap(rewardmap)
        self.bumper_obstacle_map = torch.zeros_like(rewardmap)
        self.trace["reward_map"] = rewardmap
        self.count = 0

    def get_trace(self, device="cpu") -> Dict:
        return self.trace

    def clear_trace(self):
        self.trace = {
            "reward_map": torch.zeros([1, 1, GRID_SIZE, GRID_SIZE]),
            "v_image": torch.zeros([1, 1, GRID_SIZE, GRID_SIZE]),
            "occupancy_2d": torch.zeros([1, 1, GRID_SIZE, GRID_SIZE])
        }

    def select_gridaction(self, q_function, x, y):
        #if (x, y) != self.prev_pos:
        #    self.tried_and_failed_actions = []

        q_function_0 = q_function[0:1, :, x, y]
        action_idx = q_function_0.argmax(dim=1)
        action_idx_int = action_idx.item()
        gridaction = self.vin.idx_to_gridaction(action_idx_int)

        # If the agent already tried all the actions, stop re-trying.
        if len(self.tried_and_failed_actions) >= 3:
            return gridaction
        # If the agent didn't move, and it has already tried gridaction in this position, try another action
        elif gridaction in self.tried_and_failed_actions:
            q_function[0:1, action_idx_int, x, y] -= 10.0
            return self.select_gridaction(q_function, x, y)
        else:
            return gridaction

    def _next_pos(self, x, y, gridaction):
        if gridaction == "LEFT":
            y = y - 1
        elif gridaction == "RIGHT":
            y = y + 1
        elif gridaction == "UP":
            x = x - 1
        elif gridaction == "DOWN":
            x = x + 1
        return x, y

    def log_pos(self, vingridaction, pos_x, pos_y):
        if vingridaction == "Stop":
            self.prev_pos = (0, 0)
            self.tried_and_failed_actions = []
        else:
            if (pos_x, pos_y) == self.prev_pos:
                self.tried_and_failed_actions.append(vingridaction)
                next_x, next_y = self._next_pos(pos_x, pos_y, vingridaction)
                self.bumper_obstacle_map[:, :, next_x, next_y] = 1.0
                print(f"Tried and failed: {self.tried_and_failed_actions}")
            else:
                # if the agent moved, the action is NOT a failing action
                self.tried_and_failed_actions = []
            self.prev_pos = (pos_x, pos_y)

    def has_failed(self) -> bool:
        return False

    def act(self, state_repr: AlfredSpatialStateRepr) -> AlfredAction:
        self.count += 1
        occupancy_map_2d = state_repr.get_obstacle_map_2d()
        self.vin.set_occupancy_map(occupancy_map_2d) # There's only one channel
        observability_map_2d = state_repr.get_observability_map_2d(floor_level=True)
        self.vin.set_observability_map(observability_map_2d)
        self.vin.set_extra_obstacle_map(self.bumper_obstacle_map)

        q_image = self.vin.compute_q_image()

        x_vx, y_vx, z_vx = state_repr.get_pos_xyz_vx()
        x_vx_q, y_vx_q = self.vin.remap_xy(x_vx, y_vx, state_repr.data.data.shape[2])

        if self.prev_act == "MoveAhead":
            self.log_pos(self.prev_gridact, x_vx_q, y_vx_q)

        vb, vc, vh, vw = q_image.shape
        x_vx_q = max(min(x_vx_q, vh-1), 0)
        y_vx_q = max(min(y_vx_q, vw-1), 0)
        # Mark agent position for debugging purposes
        ANNOTATE_FOR_TREE_VIZ = True
        VIZ = GLOBAL_VIZ
        v_image = q_image.max(dim=1).values
        if ANNOTATE_FOR_TREE_VIZ:
            state_repr.annotations["vin_q_image"] = q_image
            state_repr.annotations["vin_v_image"] = v_image.repeat((3, 1, 1))
            state_repr.annotations["vin_occupancy"] = occupancy_map_2d[:, 0, :, :]
            state_repr.annotations["vin_v_image"][0:2, x_vx_q, y_vx_q] = 0.0
            self.trace["v_image"] = v_image[None, :, :, :]
            self.trace["occupancy_2d"] = occupancy_map_2d
            self.trace["reward_map"] = self.vin.rewardmap
        if VIZ:
            v_image_c_neg = torch.tensor([1, 0, 0], device=v_image.device)[:, None, None] * v_image.clamp(-1000, 0) * (-1)
            v_image_c_pos = torch.tensor([0, 1, 0], device=v_image.device)[:, None, None] * v_image.clamp(0, 1000) * 1
            peak = max(v_image_c_neg.max().item(), v_image_c_pos.max().item())
            v_image_c = (v_image_c_neg + v_image_c_pos) / peak
            v_image_c[:, x_vx_q, y_vx_q] = torch.tensor([0, 0, 1])
            show_image(v_image_c, "VIN Value Image", scale=4, waitkey=1)

        gridaction = self.select_gridaction(q_image, x_vx_q, y_vx_q)
        self.prev_gridact = gridaction # Log this so that we can keep track of which MoveAhead succeed and fail

        # If reached the right place, rotate towards the object and stop
        if gridaction == "STOP":
            self.prev_act = "Stop"
            return AlfredAction("Stop", AlfredAction.get_empty_argument_mask())
        # Otherwise rotate in the direction according to the gridworld action, and then move forward
        else:
            target_yaw = {
                "UP": math.pi,              # Negative X direction
                "RIGHT": math.pi / 2,       # Positive Y direction
                "DOWN": 0,                  # Positive X direction
                "LEFT": math.pi * 3 / 2     # Negative Y direction
            }[gridaction]
            self.rotate_to_yaw_skill.set_goal(target_yaw)
            rotate_action = self.rotate_to_yaw_skill.act(state_repr)
            if rotate_action.is_stop():
                # This is needed to keep track of actions that move the agent
                self.prev_act = "MoveAhead"
                return AlfredAction("MoveAhead", AlfredAction.get_empty_argument_mask())
            else:
                self.prev_act = "Rotate"
                return rotate_action
