from typing import Dict, Union
import copy
import torch

from lgp.abcd.skill import Skill

from lgp.env.alfred.alfred_action import AlfredAction
from lgp.env.alfred.alfred_subgoal import AlfredSubgoal

from lgp.abcd.action import Action
from lgp.models.alfred.handcoded_skills.go_for import GoForSkill
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr

from lgp.models.alfred.handcoded_skills.tilt_to_pitch import TiltToPitchSkill

NOMINAL_PITCH = 0.5235988419208105


class InteractSkill(Skill):
    def __init__(self, gofor_skill : GoForSkill, explore_skill : Skill = None):
        super().__init__()
        self.gofor_skill : GoForSkill = gofor_skill
        self.explore_skill = explore_skill
        self.tilt_to_pitch = TiltToPitchSkill()

        # make sure all of these are in reset
        self.found = False
        self.wentfor = False
        self.interacted = False
        self.restored_nominal_pitch = False
        self.post_rotated = False
        self.post_rot_count = 0
        self.interaction_failed = None
        self.subgoal = None
        self.trace = {}
        self.clear_trace()

    def _reset(self):
        self.found = False
        self.wentfor = False
        self.interacted = False
        self.restored_nominal_pitch = False
        self.interaction_failed = None
        self.post_rotated = False
        self.post_rot_count = 0
        self.subgoal = None

    def start_new_rollout(self):
        self._reset()
        self.gofor_skill.start_new_rollout()
        if self.explore_skill is not None:
            self.explore_skill.start_new_rollout()
        self.tilt_to_pitch.start_new_rollout()

    def get_trace(self, device="cpu") -> Dict:
        for k,v in self.trace.items():
            self.trace[k] = v.to(device) if hasattr(v, "to") else v
        tr = {
            "gofor": self.gofor_skill.get_trace(device),
            "interact": self.trace,
        }
        if self.explore_skill is not None:
            tr["explore"] = self.explore_skill.get_trace(device)
        return tr

    def clear_trace(self):
        self.gofor_skill.clear_trace()
        if self.explore_skill is not None:
            self.explore_skill.clear_trace()
        self.trace = {
            "fpv_argument_mask": torch.zeros((1, 1, 300, 300)),
            "fpv_voxel_argument_mask": torch.zeros((1, 1, 300, 300)),
            "fpv_semantic_argument_mask": torch.zeros((1, 1, 300, 300)),
            "llc_flow_state": " "
        }

    def _fpv_trace(self, trace_stuff):
        if trace_stuff is None:
            self.trace["fpv_argument_mask"] = torch.zeros((1, 1, 300, 300))
            self.trace["fpv_voxel_argument_mask"] = torch.zeros((1, 1, 300, 300))
            self.trace["fpv_semantic_argument_mask"] = torch.zeros((1, 1, 300, 300))
        else:
            self.trace["fpv_argument_mask"] = trace_stuff["fpv_argument_mask"]
            self.trace["fpv_voxel_argument_mask"] = trace_stuff["fpv_voxel_argument_mask"]
            self.trace["fpv_semantic_argument_mask"] = trace_stuff["fpv_semantic_argument_mask"]

    def set_goal(self, subgoal: AlfredSubgoal):
        assert isinstance(subgoal, AlfredSubgoal)
        prev_goal = copy.deepcopy(self.subgoal)
        self._reset()
        self.subgoal = subgoal

        same_goal = (self.subgoal == prev_goal)
        if self.explore_skill is not None:
            self.explore_skill.set_goal(subgoal)
        self.gofor_skill.set_goal(subgoal, remember_past_failures=same_goal)
        self.tilt_to_pitch.set_goal(NOMINAL_PITCH)

    def has_failed(self) -> bool:
        if self.interaction_failed is None:
            return False
        return self.interaction_failed

    def act(self, state_repr: AlfredSpatialStateRepr) -> Action:
        # Generate action arguments at EVERY timestep, even if not interacting to better show the agent reasoning
        _, trace_stuff = self.subgoal.to_action(state_repr, state_repr.observation, return_intermediates=True)
        self._fpv_trace(trace_stuff)
        self.trace["llc_flow_state"] = "Exploring"

        # Use the "Explore" skill (if we have one) to locate the object
        if not self.found and self.explore_skill is not None:
            action : Action = self.explore_skill.act(state_repr)
            if action.is_stop():
                print(f"EXPLORE FINISHED: {self.subgoal.arg_str()}")
                self.found = True
            else:
                print(f"NOT FOUND. LOOKING FOR: {self.subgoal.arg_str()}")
                return action

        self.trace["llc_flow_state"] = "Interacting"

        # First go to a position from which this interaction action can be executed
        if not self.wentfor:
            action : Action = self.gofor_skill.act(state_repr)
            if action.is_stop():
                self.wentfor = True
            else:
                return action

        # Then execute the interaction action
        if not self.interacted:
            action, trace_stuff = self.subgoal.to_action(state_repr, state_repr.observation, return_intermediates=True)
            self._fpv_trace(trace_stuff)
            self.interacted = True
            # Try to preempt invalid interaction actions without executing them
            if action.is_valid():
                return action
            else:
                self.interaction_failed = True

        # Detect failed interactions to avoid adding this high-level action to the action history
        # Only do it once. interaction_failed can be one of {None, False, True}
        if self.interaction_failed is None:
            self.interaction_failed = state_repr.observation.last_action_error

        # Finally reset pitch to the default/nominal value with which the agent keeps navigating around
        if not self.restored_nominal_pitch:
            action = self.tilt_to_pitch.act(state_repr)
            if action.is_stop():
                self.restored_nominal_pitch = True
                self.tilt_to_pitch.set_goal(NOMINAL_PITCH)
            else:
                return action

        # Finally, execute the stop action to end the skill and revert to the high-level policy
        return AlfredAction("Stop", AlfredAction.get_empty_argument_mask())