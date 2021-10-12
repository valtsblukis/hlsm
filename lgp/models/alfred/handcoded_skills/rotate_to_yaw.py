from typing import Dict
import math

from lgp.abcd.skill import Skill
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr
from lgp.env.alfred.alfred_action import AlfredAction


class RotateToYawSkill(Skill):
    def __init__(self):
        super().__init__()
        self.target_yaw = None

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.target_yaw = None

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def set_goal(self, target_yaw):
        self._reset()
        # Rotate to 0-2pi range, and find the closest target angle
        target_yaw = target_yaw % (math.pi * 2)
        options = [0, math.pi/2, math.pi, math.pi * 3 / 2, math.pi * 2]
        dists = [math.fabs(target_yaw - o) for o in options]
        idsts = [(o, d) for o, d in zip(options, dists)]
        self.target_yaw = min(idsts, key=lambda m: m[1])[0]

    def has_failed(self) -> bool:
        return False

    def act(self, state_repr : AlfredSpatialStateRepr) -> AlfredAction:
        roll, pitch, yaw = state_repr.get_rpy()
        # Allow control error to be between -pi and +pi
        ctrl_diff = yaw - self.target_yaw
        ctrl_diff = (ctrl_diff + math.pi) % (math.pi * 2) - math.pi

        # Rotate to the correct angle
        if ctrl_diff < -1e-2:
            action_type = "RotateLeft"
        elif ctrl_diff > +1e-2:
            action_type = "RotateRight"
        else:
            action_type = "Stop"

        return AlfredAction(action_type=action_type, argument_mask=AlfredAction.get_empty_argument_mask())
