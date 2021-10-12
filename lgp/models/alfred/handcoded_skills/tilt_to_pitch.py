from typing import Dict
import math

from lgp.abcd.skill import Skill
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.env.alfred.alfred_action import AlfredAction


DEBUG_DISABLE_PITCHING = False
MIN_PITCH = -1.30899694
MAX_PITCH = 1.30899694


class TiltToPitchSkill(Skill):
    def __init__(self):
        super().__init__()
        self.target_pitch = None
        self.last_diff = None

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.target_pitch = None
        self.last_diff = None

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def set_goal(self, target_pitch):
        self.target_pitch = max(min(target_pitch, MAX_PITCH), MIN_PITCH)
        self.last_diff = None


    def has_failed(self) -> bool:
        return False

    def act(self, state_repr : AlfredSpatialStateRepr) -> AlfredAction:
        # Debug what happens if we disable pitching
        if DEBUG_DISABLE_PITCHING:
            return AlfredAction(action_type="Stop", argument_mask=AlfredAction.get_empty_argument_mask())

        pitch = state_repr.get_camera_pitch_deg()
        pitch = math.radians(pitch)

        # Allow control error to be between -pi and +pi
        ctrl_diff = pitch - self.target_pitch
        ctrl_diff = (ctrl_diff + math.pi) % (math.pi * 2) - math.pi

        step_size = math.pi * (15 / 180)

        # Rotate to the correct angle
        if ctrl_diff < -step_size/2:
            action_type = "LookDown"
        elif ctrl_diff > step_size/2:
            action_type = "LookUp"
        else:
            action_type = "Stop"

        # Sometimes tilting gets stuck due to collisions. This is to abort tilting and avoid getting stuck in an infinte loop.
        if action_type in ["LookDown", "LookUp"] and self.last_diff == ctrl_diff:
            print("TiltToPitch: Seems to be stuck. Stopping!")
            action_type = "Stop"
        self.last_diff = ctrl_diff

        return AlfredAction(action_type=action_type, argument_mask=AlfredAction.get_empty_argument_mask())
